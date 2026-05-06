import logging
import os
import re

import ale_py
from dm_control import suite
from dm_env import specs
import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces import Box
import numpy as np

from src.utils import ConfigWrapper


gym.register_envs(ale_py)


def make_env(config: ConfigWrapper) -> gym.Env:
    """
    Function to make RL environment.

    Args:
        config (ConfigWrapper): Experiment config.

    Returns:
        gym.Env: Gymnasium environment.
    """
    task_type = config.task.name[0]
    if task_type == "gym":
        env = make_gym_env(config)
    elif task_type == "dmc":
        env = make_dmc_env(config)
    elif task_type == "atari":
        env = make_atari_env(config)

    return env


def make_atari_env(config: ConfigWrapper) -> gym.Env:
    """
    Function to make ALE environment.

    Args:
        config (ConfigWrapper): Experiment config.

    Returns:
        gym.Env: ALE gymnasium environment.
    """
    env_name = config.task.name[1]
    env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    env.action_space.seed(config.seed)

    return env

### Code adopted from https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl_utils/atari_wrappers.py#L213 ###
class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        Args:
            reward (float): Reward from environment.

        Returns:
            float: Clipped reward.
        """
        return np.sign(float(reward))


def make_gym_env(config: ConfigWrapper) -> gym.Env:
    """
    Function to make MuJoCo Gym environment.

    Args:
        config (ConfigWrapper): Experiment config.

    Returns:
        gym.Env: MuJoCo gymnasium environment.
    """
    env_name = config.task.name[1]
    env = gym.make(env_name, render_mode="rgb_array")

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if "ppo" in config.experiment_type.lower() or "reinforce" in config.experiment_type.lower():
        env = gym.wrappers.ClipAction(env)

        # List of techniques from Enstrom et. al (2020)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    return env


def make_dmc_env(config: ConfigWrapper) -> gym.Env:
    """
    Function to make DMC environment.

    Args:
        config (ConfigWrapper): Experiment config.

    Returns:
        gym.Env: DMC gymnasium environment.
    """
    env_name = config.task.name[1].split('-')
    domain_name = env_name[0]
    task_name = env_name[1]
    env = DMCGym(
        domain_name,
        task_name,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if "ppo" in config.experiment_type.lower() or "reinforce" in config.experiment_type.lower():
        env = gym.wrappers.ClipAction(env)  # Required for limiting the action space

        # Techniques from Enstrom et. al (2020)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    return env

### Code from: https://github.com/imgeorgiev/dmc2gymnasium/blob/main/dmc2gymnasium/DMCGym.py ###
def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMCGym(Env):
    def __init__(
        self,
        domain,
        task,
        task_kwargs={},
        environment_kwargs={},
        rendering="egl",
        render_height=64,
        render_width=64,
        render_camera_id=0,
    ):
        """Wrapper class that aligns the interface of DMC to Mujoco."""

        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)


if __name__ == "__main__":
    from src.utils import wrap_config_dict
    config = {
        "experiment_type": "SACBasic",
        "task": {
            "name": ["dmc", "lqr-lqr_6_2"]
        }
    }
    config = wrap_config_dict(config)

    task = make_env(config)
    action_space = task.action_space
    task.reset()
    for i in range(10000):
        a = np.random.uniform(low=-1, high=1, size=action_space.shape)
        _, _, terminations, truncations, infors = task.step(a)
        done = terminations or truncations
        if done:
            print("done!")
            break
