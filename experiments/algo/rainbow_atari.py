import collections
from collections import deque
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from shared_config import Metrics
from src.tasks import make_env
from src.utils import (
    ConfigWrapper,
    StatTracker,
    duration,
    seed_everything,
    mkdir,
)

from ..experiment import Experiment


### Adopted from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py ###
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# ALGO LOGIC: initialize agent here:
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.action_space.n
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
        )
        conv_output_size = 3136

        self.value_head = nn.Sequential(NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms))

        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


PrioritizedBatch = collections.namedtuple(
    "PrioritizedBatch", ["observations", "actions", "rewards", "next_observations", "dones", "indices", "weights"]
)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, device, n_step, stack_size, gamma, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.stack_size = stack_size
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.buffer_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)
        self.buffer_dones[-1] = 1

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        # self.min_tree = MinSegmentTree(capacity)

        # For n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for i in range(len(self.n_step_buffer)):
            reward += self.gamma**i * self.n_step_buffer[i][2]
            if self.n_step_buffer[i][4]:
                next_obs = self.n_step_buffer[i][3]
                done = True
                break
        return reward, next_obs, done

    def add(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.pos
        self.buffer_obs[idx] = obs[-1, :, :]
        self.buffer_obs[(idx + 1) % self.capacity] = next_obs[-1, :, :]
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        # self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # Zero out priority for sum_tree for pos to pos + stack_size
        for i in range(self.stack_size):
            self.sum_tree.update((self.pos + i) % self.capacity, 0)

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        indices = np.array(indices)

        samples = {
            "observations": torch.from_numpy(self._sample_stacked_obs(indices)).to(self.device),
            "actions": torch.from_numpy(self.buffer_actions[indices]).to(self.device).unsqueeze(1),
            "rewards": torch.from_numpy(self.buffer_rewards[indices]).to(self.device).unsqueeze(1),
            "next_observations": torch.from_numpy(self._sample_stacked_obs((indices + 1) % self.capacity)).to(self.device),
            "dones": torch.from_numpy(self.buffer_dones[indices]).to(self.device).unsqueeze(1),
        }

        probs = np.array([self.sum_tree.tree[idx + self.capacity - 1] for idx in indices])
        weights = (self.size * probs / p_total) ** -self.beta
        weights = weights / weights.max()
        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices

        return PrioritizedBatch(**samples)

    def _sample_stacked_obs(self, indices):
        batch_size = len(indices)

        offsets = np.arange(-(self.stack_size - 1), 1)
        stack_indices = (indices[:, None] + offsets) % self.capacity
        batched_obs = self.buffer_obs[stack_indices]

        for i in range(batch_size):
            mask = self.buffer_dones[stack_indices[i]]

            if np.any(mask[:-1] == 1):
                done_indices = np.where(mask[:-1] == 1)[0]
                last_reset_pos = done_indices[-1]
                first_frame = batched_obs[i, last_reset_pos + 1]
                batched_obs[i, :last_reset_pos + 1] = first_frame

        return batched_obs

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            priority = priority**self.alpha
            self.sum_tree.update(idx, priority)
            # self.min_tree.update(idx, priority)


class RainbowAtariSave(Experiment):
    """
    Basic Rainbow.
    """
    def __init__(self, config: ConfigWrapper, root_dir: Path, session_name: str, num_resub: int=0):
        super(RainbowAtariSave, self).__init__(config, root_dir, session_name)

        ### Set seed ###
        self.config["seed"] += self.num_resub
        seed_everything(self.config.seed)

        self.task = make_env(self.config)

        self.q_network = NoisyDuelingDistributionalNetwork(
            self.task,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.q_network.parameters(), lr=self.config.step_size, eps=1.5e-4)
        self.target_network = NoisyDuelingDistributionalNetwork(
            self.task,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.task.observation_space.shape[-2:],
            self.device,
            self.config.psteps,
            self.task.observation_space.shape[0],
            self.config.gamma,
            self.config.palpha,
            self.config.pbeta,
            self.config.peps,
        )

        # Stat tracker
        self.bin_size = self.config.n_steps // self.config.n_bins
        self.binned_statistics = StatTracker(self.config.n_bins, self.save_dir)
        self.tracker = StatTracker(self.bin_size + 100, self.save_dir)

        # Training tracking
        self.t = 0
        self.epi_cnt = 0

    def save_interim(self):
        tmp_save_dir = Path(f"{os.environ['SLURM_TMPDIR']}/interim/{self.config.experiment_name}/{self.config.config_idx}")
        mkdir(tmp_save_dir)

        # Save all the pytorch objects
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optim": self.optim.state_dict(),
        }, tmp_save_dir / "interim_modules.pt")

        torch.save({
            "pos": self.rb.pos,
            "size": self.rb.size,
            "max_priority": self.rb.max_priority,
            "n_step_buffer": self.rb.n_step_buffer,
            "t": self.t,
            "epi_cnt": self.epi_cnt,
            "binned_indices": self.binned_statistics.indices,
            "tracker_indices": self.tracker.indices,
        }, tmp_save_dir / "interim_vars.pt")

        # Save all the numpy objects
        np.savez_compressed(
            tmp_save_dir / "interim.npz",
            buffer_obs=self.rb.buffer_obs,
            buffer_actions=self.rb.buffer_actions,
            buffer_rewards=self.rb.buffer_rewards,
            buffer_dones=self.rb.buffer_dones,
            sum_tree=self.rb.sum_tree.tree,
            binned_episodic_returns=self.binned_statistics.results[f"{Metrics.episodic_return}_mean"],
            tracker_episodic_returns=self.tracker.results[Metrics.episodic_return],
        )

    def load_interim(self):
        tmp_save_dir = Path(f"{os.environ['SLURM_TMPDIR']}/interim/{self.config.experiment_name}/{self.config.config_idx}")
        mkdir(tmp_save_dir)

        # Load pytorch objects
        ckpt_modules = torch.load(tmp_save_dir / "interim_modules.pt")
        self.q_network.load_state_dict(ckpt_modules["q_network"])
        self.target_network.load_state_dict(ckpt_modules["target_network"])
        self.optim.load_state_dict(ckpt_modules["optim"])

        ckpt_vars = torch.load(tmp_save_dir / "interim_vars.pt")
        self.rb.pos = ckpt_vars["pos"]
        self.rb.size = ckpt_vars["size"]
        self.rb.max_priority = ckpt_vars["max_priority"]
        self.rb.n_step_buffer = ckpt_vars["n_step_buffer"]
        self.t = ckpt_vars['t']
        self.epi_cnt = ckpt_vars["epi_cnt"]
        self.binned_statistics.indices = ckpt_vars["binned_indices"]
        self.tracker.indices = ckpt_vars["tracker_indices"]

        # Load numpy objects
        np_arrays = np.load(tmp_save_dir / "interim.npz")
        self.rb.buffer_obs = np_arrays["buffer_obs"]
        self.rb.buffer_actions = np_arrays["buffer_actions"]
        self.rb.buffer_rewards = np_arrays["buffer_rewards"]
        self.rb.buffer_dones = np_arrays["buffer_dones"]
        self.rb.sum_tree.tree = np_arrays["sum_tree"]
        self.binned_statistics.results[f"{Metrics.episodic_return}_mean"] = np_arrays["binned_episodic_returns"]
        self.tracker.results[Metrics.episodic_return] = np_arrays["tracker_episodic_returns"]

    def run_experiment(self):
        if self.num_resub > 0:
            self.load_interim()
        env_seed = np.random.randint(low=0, high=int(1e6))
        obs, _ = self.task.reset(seed=env_seed)
        st_time = time.time()
        while self.t < self.config.n_steps:
            # Progress timestep
            self.t += 1

            self.rb.beta = min(
                1.0, self.config.pbeta + (self.t - 1) * (1.0 - self.config.pbeta) / self.config.psteps
            )

            with torch.no_grad():
                q_dist = self.q_network(torch.Tensor(obs).unsqueeze(0).to(self.device))
                q_values = torch.sum(q_dist * self.q_network.support, dim=2)
                action = torch.argmax(q_values).cpu().numpy()

            next_obs, reward, termination, truncation, infos = self.task.step(action)

            self.rb.add(obs, action, reward, next_obs, termination)

            if termination or truncation:
                self.tracker.add({Metrics.episodic_return: [infos["episode"]['r']]})
                obs, _ = self.task.reset()
                self.epi_cnt += 1
            else:
                obs = next_obs

            # Aggregate episodic return per bin on the fly
            if self.t % self.bin_size == 0:
                tracker_idx = self.tracker.indices[Metrics.episodic_return]
                self.binned_statistics.add({
                    f"{Metrics.episodic_return}_mean": [np.apply_along_axis(np.mean, 0, self.tracker.results[Metrics.episodic_return][:tracker_idx])]
                })
                self.tracker.reset([Metrics.episodic_return])

            if self.t % 10_000_000 == 0:
                hrs, mins, secs = duration(st_time)
                print(self.t, hrs, mins, secs)

            if self.t > self.config.learning_starts:
                if self.t % self.config.train_frequency == 0:
                    # Reset noise for networks
                    self.q_network.reset_noise()
                    self.target_network.reset_noise()

                    # Sample from replay buffer
                    data = self.rb.sample(self.config.batch_size)

                    with torch.no_grad():
                        next_dist = self.target_network(data.next_observations)
                        support = self.target_network.support

                        # Double q learning
                        next_dist_online = self.q_network(data.next_observations)
                        next_q_online = torch.sum(next_dist_online * support, dim=2)
                        best_actions = torch.argmax(next_q_online, dim=1)
                        next_pmfs = next_dist[torch.arange(self.config.batch_size), best_actions]

                        # n-step Bellman update
                        gamma_n = self.config.gamma**self.config.psteps
                        next_atoms = data.rewards + gamma_n * support * (1 - data.dones.float())
                        tz = next_atoms.clamp(self.q_network.v_min, self.q_network.v_max)

                        # Projection
                        delta_z = self.q_network.delta_z
                        b = (tz - self.q_network.v_min) / delta_z
                        l = b.floor().clamp(0, self.config.n_atoms - 1)
                        u = b.ceil().clamp(0, self.config.n_atoms - 1)
                        d_m_l = (u.float() + (l == b).float() - b) * next_pmfs
                        d_m_u = (b - l) * next_pmfs

                        target_pmfs = torch.zeros_like(next_pmfs)
                        for i in range(target_pmfs.size(0)):
                            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                    dist = self.q_network(data.observations)
                    pred_dist = dist.gather(1, data.actions.unsqueeze(-1).expand(-1, -1, self.config.n_atoms)).squeeze(1)
                    log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

                    loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
                    loss = (loss_per_sample * data.weights.squeeze()).mean()

                    # Update priorities
                    new_priorities = loss_per_sample.detach().cpu().numpy()
                    self.rb.update_priorities(data.indices, new_priorities)

                    # Optimization step
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                if self.t % self.config.target_update_frequency == 0:
                    for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_param.data.copy_(
                            self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                        )

            hrs, _, _ = duration(st_time)
            if hrs >= 23:
                self.save_interim()
                exit(124)

        # Save results
        self.binned_statistics.save_non_empty()
        hours, mins, secs = duration(st_time)
        print(f"Experiment {self.session_name} ran for {self.t} steps, {self.epi_cnt} episodes ({hours} hrs {mins} mins {secs} secs)")
        exit(0)
