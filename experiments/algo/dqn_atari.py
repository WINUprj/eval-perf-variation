from abc import ABC, abstractmethod
from pathlib import Path
import psutil
import random
import time
from typing import Any, NamedTuple

import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from shared_config import Metrics
from src.tasks import make_env
from src.utils import (
    ConfigWrapper,
    StatTracker,
    seed_everything,
    duration,
)

from ..experiment import Experiment


### Adopted from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py ###
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
):
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: torch.device) -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device,
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray):
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device,
        num_stacks: int = 4,
    ):
        super().__init__(buffer_size, observation_space, action_space, device)

        # Adjust buffer size
        self.buffer_size = buffer_size
        self.num_stacks = num_stacks

        # Only store most recent frame to the replay buffer
        self.obs_shape = self.obs_shape[-2:]

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.timeouts = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones[-1] = 1.

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        obs = obs[-1, :, :].reshape(*self.obs_shape)
        next_obs = next_obs[-1, :, :].reshape(*self.obs_shape)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        # Do not sample the element with index `self.pos` as the transitions is invalid
        if self.full:
            batch_inds = (np.random.randint(self.num_stacks, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        stacked_observations = self._sample_stacked_obs(batch_inds)
        stacked_next_observations = self._sample_stacked_obs((batch_inds + 1) % self.buffer_size)

        data = (
            stacked_observations,
            self.actions[batch_inds, :],
            stacked_next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _sample_stacked_obs(self, batch_inds: np.ndarray):
        batch_size = len(batch_inds)

        offsets = np.arange(-(self.num_stacks - 1), 1)
        stack_indices = (batch_inds[:, None] + offsets) % self.buffer_size
        batched_obs = self.observations[stack_indices]
        done_mask = self.dones * (1 - self.timeouts)

        for i in range(batch_size):
            mask = done_mask[stack_indices[i]]

            if np.any(mask[:-1] == 1):
                done_indices = np.where(mask[:-1] == 1)[0]
                last_reset_pos = done_indices[-1]
                first_frame = batched_obs[i, last_reset_pos + 1]
                batched_obs[i, :last_reset_pos + 1] = first_frame

        return batched_obs

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNAtari(Experiment):
    """
    Basic DQN.
    """
    def __init__(self, config: ConfigWrapper, root_dir: Path, session_name: str):
        super(DQNAtari, self).__init__(config, root_dir, session_name)

        ### Set seed ###
        seed_everything(self.config.seed)

        # Setup task
        self.task = make_env(self.config)

        # Network initialization
        self.q_network = QNetwork(self.task).to(self.device)
        self.q_network.compile(mode="reduce-overhead")
        self.optim = torch.optim.Adam(self.q_network.parameters(), lr=self.config.step_size)
        self.target_network = QNetwork(self.task).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # ReplayBuffer
        self.rb = ReplayBuffer(
            self.config.buffer_size,
            self.task.observation_space,
            self.task.action_space,
            self.device,
            num_stacks=4,
        )

        # Misc Hyperparameters
        self.epsilon = 1

        # Stat tracker
        self.bin_size = self.config.n_steps // self.config.n_bins
        self.binned_statistics = StatTracker(self.config.n_bins, self.save_dir)
        self.tracker = StatTracker(self.bin_size + 100, self.save_dir)

    def _take_action(self, obs):
        if random.random() < self.epsilon:
            action = self.task.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.Tensor(obs).unsqueeze(0).to(self.device))
                action = torch.argmax(q_values).cpu().numpy()

        return action

    def run_experiment(self):
        env_seed = np.random.randint(low=0, high=int(1e6))
        obs, _ = self.task.reset(seed=env_seed)
        t = 0
        epi_cnt = 0
        st_time = time.time()
        while t < self.config.n_steps:
            # Progress timestep
            t += 1

            # Epsilon
            self.epsilon = linear_schedule(self.config.start_e, self.config.end_e, self.config.exploration_fraction * self.config.n_steps, t-1)

            # Act
            action = self._take_action(obs)
            next_obs, reward, termination, truncation, infos = self.task.step(action)

            self.rb.add(obs, next_obs, action, reward, termination, infos)

            if truncation or termination:
                self.tracker.add({Metrics.episodic_return: [infos["episode"]['r']]})
                obs, _ = self.task.reset()
                epi_cnt += 1
            else:
                obs = next_obs

            # Aggregate episodic return per bin on the fly
            if t % self.bin_size == 0:
                tracker_idx = self.tracker.indices[Metrics.episodic_return]
                self.binned_statistics.add({
                    f"{Metrics.episodic_return}_mean": [np.apply_along_axis(np.mean, 0, self.tracker.results[Metrics.episodic_return][:tracker_idx])]
                })
                self.tracker.reset([Metrics.episodic_return])

            if t > self.config.learning_starts:
                if t % self.config.train_frequency == 0:
                    data = self.rb.sample(self.config.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        target = data.rewards.flatten() + self.config.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(old_val, target)

                    # Optimization step
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                if t % self.config.target_update_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_network_param.data.copy_(
                            self.config.tau * q_network_param.data + (1.0 - self.config.tau) * target_network_param.data
                        )
            if t % 1_000_000 == 0:
                hours, mins, secs = duration(st_time)
                print(t, hours, mins, secs)

        # Save results
        self.binned_statistics.save_non_empty()
        hours, mins, secs = duration(st_time)
        print(f"Experiment {self.session_name} ran for {t} steps, {epi_cnt} episodes ({hours} hrs {mins} mins {secs} secs)")
