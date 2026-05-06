import datetime
from pathlib import Path
import time
from typing import Union

import torch
from torch.nn import functional as F
import numpy as np

from shared_config import Metrics
import src
from src.tasks import make_env
from src.utils import (
    ConfigWrapper,
    Logger,
    ReplayBuffer,
    StatTracker,
    duration,
    frequency,
    seed_everything,
    negative_frequency,
)

from ..experiment import Experiment


# ---------- SAC-related components ---------- #
class SACActor(nn.Module):
    def __init__(self, env, actor_config, pnorm=False):
        super(SACActor, self).__init__()

        # Network configurations
        encoder_arch_name = actor_config.encoder.arch_name
        encoder_kwargs = actor_config.encoder.kwargs
        if "layer_sizes" in encoder_kwargs:
            encoder_kwargs["layer_sizes"][0] = np.array(env.observation_space.shape).prod()
            encoder_out_shape = encoder_kwargs.layer_sizes[-1]
        elif "in_units" in encoder_kwargs:
            encoder_kwargs["in_units"] = np.array(env.observation_space.shape).prod()
            encoder_out_shape = encoder_kwargs.hidden_units
        self.encoder = getattr(src.networks, encoder_arch_name)(**encoder_kwargs)
        self.fc_mean = nn.Linear(encoder_out_shape, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(encoder_out_shape, np.prod(env.action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

        self.log_std_min = actor_config.log_std_min
        self.log_std_max = actor_config.log_std_max

        self.pnorm = pnorm

    def get_features(self, x):
        x = self.encoder(x)
        if self.pnorm:
            x = F.normalize(x, dim=-1)

        return x

    def forward(self, x):
        x = self.get_features(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def act(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, normal.entropy().sum(-1), std


class SACCritic(nn.Module):
    def __init__(self, env, critic_config, pnorm=False):
        super(SACCritic, self).__init__()

        # Encoder configs
        encoder_arch_name = critic_config.encoder.arch_name
        encoder_kwargs = critic_config.encoder.kwargs
        if "layer_sizes" in encoder_kwargs:
            encoder_kwargs["layer_sizes"][0] = np.array(env.observation_space.shape).prod() + np.array(env.action_space.shape).prod()
        elif "in_units" in encoder_kwargs:
            encoder_kwargs["in_units"] = np.array(env.observation_space.shape).prod() + np.array(env.action_space.shape).prod()

        # Predictor configs
        predictor_arch_name = critic_config.predictor.arch_name
        predictor_kwargs = critic_config.predictor.kwargs

        # Create a pair of networks
        self.q1_encoder = getattr(src.networks, encoder_arch_name)(**encoder_kwargs)
        self.q1_predictor = getattr(src.networks, predictor_arch_name)(**predictor_kwargs)

        self.q2_encoder = getattr(src.networks, encoder_arch_name)(**encoder_kwargs)
        self.q2_predictor = getattr(src.networks, predictor_arch_name)(**predictor_kwargs)

        self.pnorm = pnorm

    def get_features(self, obs, action):
        x = torch.cat([obs, action], 1)

        q1_val_enc = self.q1_encoder(x)
        if self.pnorm:
            q1_val_enc = F.normalize(q1_val_enc, dim=-1)

        q2_val_enc = self.q2_encoder(x)
        if self.pnorm:
            q2_val_enc = F.normalize(q2_val_enc, dim=-1)

        return q1_val_enc, q2_val_enc

    def forward(self, obs, action):
        q1_val_enc, q2_val_enc = self.get_features(obs, action)

        q1_val = self.q1_predictor(q1_val_enc)
        q2_val = self.q2_predictor(q2_val_enc)

        return q1_val, q2_val


class SAC(Experiment):
    """
    Basic SAC code with minimal statistics.

    Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
    """
    def __init__(self, config: ConfigWrapper, root_dir: Path, session_name: str):
        super(SAC, self).__init__(config, root_dir, session_name)

        ### Set seed ###
        seed_everything(self.config.seed)

        # Sub-configs
        actor_config = self.config.learner.actor
        critic_config = self.config.learner.critic
        actor_optim_config = self.config.optim.actor
        critic_optim_config = self.config.optim.critic
        rb_config = self.config.replay_buffer

        # Initialize task
        if self.config.env_change:
            self.env_seed = self.config.config_idx
        elif self.config.net_change:
            self.env_seed = 1
        else:
            self.env_seed = np.random.randint(low=0, high=int(1e6))

        self.config["env_seed"] = self.env_seed
        self.task = make_env(self.config)

        # Initialize normalizer if needed
        self.pnorm = self.config.learner.pnorm

        # Initialize all the network architectures
        self.actor = SACActor(self.task, actor_config, self.pnorm).to(self.device)
        self.qf = SACCritic(self.task, critic_config, self.pnorm).to(self.device)
        self.qf_target = SACCritic(self.task, critic_config, self.pnorm).to(self.device)
        self.qf_target.load_state_dict(self.qf.state_dict())

        # Store critic parameters separately
        self.q1_names = [name for name, _ in self.qf.named_parameters() if "q1_" in name]
        self.q2_names = [name for name, _ in self.qf.named_parameters() if "q2_" in name]
        self.q1_params = [param for name, param in self.qf.named_parameters() if "q1_" in name]
        self.q2_params = [param for name, param in self.qf.named_parameters() if "q2_" in name]

        ### Optimizers ###
        self.step_size = self.config.optim.step_size
        if "lr" not in actor_optim_config["kwargs"]:
            actor_optim_config["kwargs"]["lr"] = self.step_size
        if "lr" not in critic_optim_config["kwargs"]:
            critic_optim_config["kwargs"]["lr"] = self.step_size

        optim_prefix = "Custom"
        actor_optim_name = optim_prefix + self.config.optim.actor.name
        self.actor_opt = getattr(src.optimizers, actor_optim_name)(
            list(self.actor.parameters()),
            default_lr=self.step_size,
        )
        self.actor_instant_counter = 0

        critic_optim_name = optim_prefix + self.config.optim.critic.name
        self.q1_opt = getattr(src.optimizers, critic_optim_name)(
            self.q1_params,
            default_lr=self.step_size,
        )
        self.q2_opt = getattr(src.optimizers, critic_optim_name)(
            self.q2_params,
            default_lr=self.step_size,
        )

        # Replay buffer
        assert self.config.n_steps <= rb_config.size, "Replay buffer must have at least the size of number of total steps."
        self.rb_keys = ["obs", "action", "reward", "next_obs", "done"]
        self.rb = ReplayBuffer(rb_config.size, self.rb_keys)

        # Misc parameters
        self.action_repetition = self.config.action_repetition
        self.n_updates = self.config.n_updates
        self.init_temperature = self.config.init_temperature
        self.sample_size = self.config.sample_size
        self.action_space = self.task.action_space
        self.learning_starts = self.config.learning_starts
        self.gamma = self.config.gamma
        self.policy_freq = self.config.policy_freq
        self.target_network_freq = self.config.target_network_freq
        self.tau = self.config.tau

        ### Adjustable temperature ###
        self.target_entropy = -torch.prod(torch.Tensor(self.task.action_space.shape).to(self.device)).item()
        if self.config.learner.actor.encoder == "SimBa":
            self.target_entropy /= 2
        self.log_alpha = torch.tensor([np.log(self.init_temperature)], requires_grad=True, device=self.device)
        self.temperature = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_optim_config["kwargs"]["lr"])

        ### Stat tracker and logger ###
        self.bin_size = self.config.n_steps // self.config.n_bins
        self.tracker = StatTracker(self.bin_size + 100, self.save_dir)
        self.binned_statistics = StatTracker(self.config.n_bins + 5, self.save_dir)
        self.logger = Logger("experiment_log", self.save_dir)

    def get_critic_target(self, next_obs: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        """Compute the target value for the critic loss with current actor and target network."""
        # Compute the value target estimates
        with torch.no_grad():
            next_obs_actions, next_obs_log_pi, _, _, _ = self.actor.act(next_obs)
            q1_next_target, q2_next_target = self.qf_target(next_obs, next_obs_actions)
            min_qf_next_target = torch.min(q1_next_target, q2_next_target) - self.temperature * next_obs_log_pi
            next_q_val = reward + self.gamma * (1 - done) * min_qf_next_target

        return next_q_val

    def _get_action(self, t: int, obs: torch.Tensor):
        """Sample action either from the uniform distribution or actor."""
        if t < self.learning_starts:
            action = np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.shape)
            return action, None, None
        else:
            with torch.no_grad():
                action, _, mean, _, std = self.actor.act(torch.Tensor(obs).to(self.device))

            action = action.detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            return action, mean, std

    def _get_value(self, obs: torch.Tensor, action: torch.Tensor):
        """Get value estimate from the Q-function."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(torch.float32)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(torch.float32)

        if len(obs.size()) == 1:
            obs = torch.unsqueeze(obs, dim=0)
        if len(action.size()) == 1:
            action = torch.unsqueeze(action, dim=0)

        with torch.no_grad():
            q1, q2 = self.qf(obs.to(self.device), action.to(self.device))
            value = torch.min(q1, q2)

        value = value.detach().cpu().numpy()
        return value

    def actor_loss(self, obs: torch.Tensor, epsilon: Union[float, None]=None):
        """Actor loss for SAC."""
        # Compute action
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        if epsilon is None:
            epsilon = torch.distributions.utils._standard_normal(mean.shape, dtype=mean.dtype, device=mean.device)
        x_t = mean + epsilon * std
        y_t = torch.tanh(x_t)
        pi = y_t * self.actor.action_scale + self.actor.action_bias

        # Compute log-probability of the action
        normal = torch.distributions.Normal(mean, std)
        log_pi = normal.log_prob(x_t)
        log_pi -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_pi = log_pi.sum(1, keepdim=True)

        # Compute actor loss
        q1_val, q2_val = self.qf(obs, pi)
        min_qf_pi = torch.min(q1_val, q2_val)
        actor_loss = ((self.temperature * log_pi) - min_qf_pi).mean()

        return actor_loss, log_pi.mean(), epsilon

    def critic_loss(self, obs: torch.Tensor, actions: torch.Tensor, next_q_val: torch.Tensor):
        """Critic loss for SAC."""
        # Compute the losses
        q1_val, q2_val = self.qf(obs, actions)

        qf1_loss = F.mse_loss(q1_val, next_q_val)
        qf2_loss = F.mse_loss(q2_val, next_q_val)
        qf_loss = qf1_loss + qf2_loss

        return qf_loss

    def update_actor(self, obs: torch.Tensor):
        """Method to update the actor parameters."""
        # Update actor
        actor_loss, entropy_reg, epsilon = self.actor_loss(obs)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        ### Update alpha ###
        with torch.no_grad():
            _, log_pi, _, _, _ = self.actor.act(obs)
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.temperature = self.log_alpha.exp().item()

        iter_data = {
            Metrics.actor_loss: [actor_loss.detach().cpu().item()],
        }

        self.tracker.add(iter_data)

    def update_critic(self, obs: torch.Tensor, actions: torch.Tensor, next_q_val: torch.Tensor):
        """Method to update the critic parameters."""
        # Copy parameters and increment the udpate counter
        qf_loss = self.critic_loss(obs, actions, next_q_val)

        # Optimize models
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        qf_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        iter_data = {
            Metrics.critic_loss: [qf_loss.detach().cpu().item()],
        }

        self.tracker.add(iter_data)

    def update(self, t: int):
        """Update both actor and critic parameters.."""
        # Sample batch data
        data = self.rb.sample(self.sample_size)
        next_obs = data["next_obs"].to(self.device)
        obs = data["obs"].to(self.device)
        action = data["action"].to(self.device)
        reward = data["reward"].to(self.device)
        done = data["done"].to(self.device)

        next_q_val = self.get_critic_target(next_obs, reward, done)

        # Update all the networks
        self.update_critic(obs, action, next_q_val)
        if t % self.policy_freq == 0:
            for _ in range(self.policy_freq):
                self.update_actor(obs)

        # soft update on target networks
        if t % self.target_network_freq == 0:
            for p, p_target in zip(self.qf.parameters(), self.qf_target.parameters()):
                p_target.data.copy_(self.tau * p.data + (1 - self.tau) * p_target.data)

    def run_experiment(self):
        self.logger.write({"start": f"{datetime.datetime.now()}"})
        obs, _ = self.task.reset(seed=self.env_seed)
        done = False
        st_time = time.time()
        epi_cnt = 0
        t = 0
        while t < self.config.n_steps:
            # Increment timestep
            t += 1

            # Get action
            action, mean, std = self._get_action(t, obs)
            if len(action.shape) >= 2 and action.shape[0] == 1:
                action = action.squeeze(0)

            # Get values and progress the environment
            reward = 0.
            for _ in range(self.action_repetition):
                next_obs, r, terminations, truncations, infos = self.task.step(action)
                done = terminations or truncations
                reward += r
                if done:
                    break

            # Store the trajectory
            store_data = {k: v for k, v in zip(self.rb_keys, [obs, action, reward, next_obs, done])}
            self.rb.register(store_data)

            # Update observation to the next observation
            obs = next_obs

            # Learn if there is sufficient amount of data
            if t > self.learning_starts:
                for _ in range(self.n_updates):
                    self.update(t)

            if done:
                # Summary data for the episode
                episodic_data = {
                    Metrics.episode_length: [infos["episode"]['l']],
                    Metrics.episodic_return: [infos["episode"]['r']],
                }

                if epi_cnt % 200 == 0:
                    self.logger.write({"Episode": epi_cnt, "Episode info": episodic_data})
                    self.logger.write(episodic_data)

                self.tracker.add(episodic_data)

                # Reset observations
                obs, _ = self.task.reset()
                done = False
                epi_cnt += 1

            if t % self.bin_size == 0:
                # Bin everything in the stat tracker
                binned_data = {}
                for key in self.tracker.results.keys():
                    tracker_idx = self.tracker.indices[key]
                    value = self.tracker.results[key][:tracker_idx]
                    if "inner_prod" in key:
                        aggregations = [np.mean, negative_frequency]
                    elif "cut_rate" in key:
                        aggregations = [frequency]
                    else:
                        aggregations = [np.mean]

                    for agg in aggregations:
                        binned_data[f"{key}_{agg.__name__}"] = [np.apply_along_axis(agg, 0, value)]

                # Save the binned data and reset the tracker
                self.binned_statistics.add(binned_data)
                self.tracker.reset(self.tracker.results.keys())

            if t % (self.config.n_steps // 10) == 0:
                hours, mins, secs = duration(st_time)
                msg = f"Timestep {t} on session {self.session_name}, seed {self.config.seed}: {hours} hrs {mins} mins {secs} sec."
                print(msg)
                self.logger.write({f"{t}": msg})

            t += self.action_repetition

        self.binned_statistics.save_non_empty()
        hours, mins, secs = duration(st_time)
        msg = f"Experiment {self.session_name} ran for {t} steps, {epi_cnt} episodes ({hours} hrs {mins} mins {secs} sec.)."
        print(msg)
        self.logger.write({"summary": msg})
        self.logger.write({"end": f"{datetime.datetime.now()}"})

        self.logger.save()
