import datetime
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F

from shared_config import Metrics
import src
from src.tasks import make_env
from src.utils import (
    ConfigWrapper,
    Logger,
    ReplayBuffer,
    StatTracker,
    duration,
    seed_everything,
    negative_frequency,
)

from ..experiment import Experiment


### Adopted from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py ###
# ---------- PPO-related components ---------- #
class PPOActor(nn.Module):
    def __init__(self, env, actor_config, pnorm=False) -> None:
        super(PPOActor, self).__init__()

        # Encoder
        encoder_arch_name = actor_config.encoder.arch_name
        encoder_kwargs = actor_config.encoder.kwargs
        if "layer_sizes" in encoder_kwargs:
            encoder_kwargs["layer_sizes"][0] = np.array(env.observation_space.shape).prod()
        elif "in_units" in encoder_kwargs:
            encoder_kwargs["in_units"] = np.array(env.observation_space.shape).prod()
        self.encoder = getattr(src.networks, encoder_arch_name)(**encoder_kwargs)

        # Predictor
        predictor_arch_name = actor_config.predictor.arch_name
        predictor_kwargs = actor_config.predictor.kwargs
        predictor_kwargs["layer_sizes"][-1] = np.array(env.action_space.shape).prod()
        self.predictor = getattr(src.networks, predictor_arch_name)(**predictor_kwargs)

        # Standard deviation estimator
        self.actor_std = nn.Parameter(torch.zeros(1, np.array(env.action_space.shape).prod()))

        # Pnorm
        self.pnorm = pnorm

    def get_features(self, x):
        x = self.encoder(x)
        if self.pnorm:
            x = F.normalize(x, dim=-1)

        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.predictor(x)

        return x

    def get_mean_std(self, obs):
        action_mean = self(obs)
        action_logstd = self.actor_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        return action_mean, action_std

    def act(self, obs):
        action_mean, action_std = self.get_mean_std(obs)
        dist = Normal(action_mean, action_std)
        action = dist.sample()

        return action, dist.log_prob(action).sum(1), action_mean, action_std

    def eval_action(self, obs, action):
        action_mean, action_std = self.get_mean_std(obs)
        dist = Normal(action_mean, action_std)

        return dist.log_prob(action).sum(1), dist.entropy().sum(1)


class PPOCritic(nn.Module):
    def __init__(self, env, critic_config, pnorm=False):
        super(PPOCritic, self).__init__()

        encoder_arch_name = critic_config.encoder.arch_name
        encoder_kwargs = critic_config.encoder.kwargs
        if "layer_sizes" in encoder_kwargs:
            encoder_kwargs["layer_sizes"][0] = np.array(env.observation_space.shape).prod()
        elif "in_units" in encoder_kwargs:
            encoder_kwargs["in_units"] = np.array(env.observation_space.shape).prod()
        self.encoder = getattr(src.networks, encoder_arch_name)(**encoder_kwargs)

        predictor_arch_name = critic_config.predictor.arch_name
        predictor_kwargs = critic_config.predictor.kwargs
        self.predictor = getattr(src.networks, predictor_arch_name)(**predictor_kwargs)

        self.pnorm = pnorm

    def get_features(self, x):
        x = self.encoder(x)
        if self.pnorm:
            x = F.normalize(x, dim=-1)

        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.predictor(x)
        return x


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)


class PPO(Experiment):
    """
    Basic PPO code with minimal statistics.

    Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
    """
    def __init__(self, config: ConfigWrapper, root_dir: Path, session_name: str):
        super(PPO, self).__init__(config, root_dir, session_name)

        ### Set seed ###
        seed_everything(self.config.seed)

        # Extract sub-configs
        learner_config = self.config.learner
        actor_optim_config = self.config.optim.actor
        critic_optim_config = self.config.optim.critic

        # Initialize the task
        self.task = make_env(self.config)
        self.pnorm = learner_config.pnorm

        ### Initialize actor and critic networks ###
        self.critic = PPOCritic(self.task, learner_config.critic, self.pnorm)
        self.critic.apply(layer_init)

        self.actor = PPOActor(self.task, learner_config.actor, self.pnorm)
        self.actor.apply(layer_init)

        ### Optimizers ###
        self.step_size = self.config.optim.step_size
        actor_optim_config["kwargs"]["lr"] = self.step_size
        critic_optim_config["kwargs"]["lr"] = self.step_size

        self.actor_opt = getattr(torch.optim, self.config.optim.actor.name)(
            list(self.actor.parameters()),
            lr=self.step_size,
        )

        self.critic_opt = getattr(torch.optim, self.config.optim.critic.name)(
            list(self.critic.parameters()),
            lr=self.step_size,
        )

        # Flag whether the parameter update uses Adam (so one can measure inner product between increment and gradient)
        self.measure_update_inner_actor = "adam" in self.config.optim.actor.name.lower()
        self.measure_update_inner_critic = "adam" in self.config.optim.critic.name.lower()

        ### Initialize replay buffer ###
        self.update_freq = self.config.update_freq
        self.rb_keys = ["obs", "logprob", "action", "value", "reward", "done"]
        self.rb = ReplayBuffer(min(self.update_freq + 10, 1_000_000), self.rb_keys)

        ### Stat tracker and logger ###
        self.bin_size = self.config.n_steps // self.config.n_bins
        self.tracker = StatTracker(self.bin_size + 100, self.save_dir)
        self.binned_statistics = StatTracker(self.config.n_bins + 5, self.save_dir)
        self.logger = Logger("experiment_log", self.save_dir)

        ### GAE configs ###
        self.gae_gamma = self.config.gae_gamma
        self.gae_lambda = self.config.gae_lambda

        ### Misc parameters ###
        self.n_epochs = self.config.n_epochs
        self.minibatch_size = self.config.minibatch_size
        self.clip_eps = self.config.clip_eps
        self.value_coef = self.config.value_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.entropy_coef = self.config.entropy_coef

    def _gae(self, rb_data: dict, obs: torch.Tensor, done: bool):
        """GAE for parameter updates.
        """
        # Estimate advantage by GAE
        with torch.no_grad():
            n = len(rb_data["obs"])
            advantage = torch.zeros((n, 1))
            cur_gae = 0
            for i in reversed(range(n)):
                if i == n-1:
                    term_gate = 1 - done
                    nxt_value = self.critic(obs)
                else:
                    term_gate = 1 - rb_data["done"][i+1]
                    nxt_value = rb_data["value"][i+1]

                # Compute advatnage in single step
                delta = rb_data["reward"][i] + self.gae_gamma * nxt_value * term_gate - rb_data["value"][i]
                cur_gae = delta + self.gae_gamma * self.gae_lambda * term_gate * cur_gae
                advantage[i] = cur_gae

            # Compute estimated returns from the advantage
            estim_returns = advantage + rb_data["value"]

        return advantage, estim_returns

    def actor_loss(
        self,
        pred_logprob: torch.Tensor,
        b_logprob: torch.Tensor,
        b_advantage: torch.Tensor,
        mean_entropy: torch.Tensor,
    ):
        """Compute actor loss for PPO."""
        # Compute policy ratio
        ratio = (pred_logprob - b_logprob).exp()

        # Actor loss
        if b_advantage.shape[0] > 1:
            normalized_b_advantage = (b_advantage - b_advantage.mean()) / (b_advantage.std() + 1e-8)
        else:
            normalized_b_advantage = b_advantage
        actor_loss_unclipped = ratio * normalized_b_advantage
        actor_loss_clipped = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * normalized_b_advantage
        actor_loss = -torch.min(actor_loss_unclipped, actor_loss_clipped).mean() - self.entropy_coef * mean_entropy

        return actor_loss
        # return -actor_loss_unclipped.mean()

    def critic_loss(
        self,
        pred_value: torch.Tensor,
        b_estim_returns: torch.Tensor,
        b_value: torch.Tensor,
    ):
        """Compute clipped critic loss for PPO."""
        assert pred_value.shape == b_estim_returns.shape
        critic_loss_unclipped = (pred_value - b_estim_returns)**2
        value_clipped = b_value + torch.clamp(pred_value - b_value, -self.clip_eps, self.clip_eps)
        critic_loss_clipped = (value_clipped - b_estim_returns)**2
        critic_loss = self.value_coef * 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

        return critic_loss
        # return critic_loss_unclipped.mean()

    def loss(
        self,
        pred_logprob: torch.Tensor,
        pred_value: torch.Tensor,
        b_logprob: torch.Tensor,
        b_value: torch.Tensor,
        b_advantage: torch.Tensor,
        b_estim_returns: torch.Tensor,
        mean_entropy: torch.Tensor,
    ):
        """Compute overall loss for PPO. Also returns individual losses."""
        # Actor loss
        actor_loss = self.actor_loss(pred_logprob, b_logprob, b_advantage, mean_entropy)

        # Critic loss (clipped)
        critic_loss = self.critic_loss(pred_value, b_estim_returns, b_value)

        # Compute overall loss
        loss = actor_loss + critic_loss

        return loss, actor_loss, critic_loss

    def update(self, obs: torch.Tensor, done: bool, t: int):
        """Update both actor and critic networks."""
        # Compute the advantage with GAE
        rb_data = self.rb.get_data(len(self.rb))
        advantage, estim_returns = self._gae(rb_data, obs, done)

        # Update the parameters for multiple epochs
        n_minibatches = (self.update_freq - 1) // self.minibatch_size + 1
        batch_indices = np.arange(len(self.rb))
        metric_data = {
            Metrics.critic_loss: [],
            Metrics.actor_loss: [],
        }

        ### Start update iterations ###
        for _ in range(self.n_epochs):
            np.random.shuffle(batch_indices)

            avg_critic_loss, avg_actor_loss = 0, 0
            for st in range(0, self.update_freq, self.minibatch_size):
                idx = batch_indices[st:min(st+self.minibatch_size, self.update_freq)]

                b_obs = rb_data["obs"][idx].to(self.device)
                b_action = rb_data["action"][idx].to(self.device)
                b_logprob = rb_data["logprob"][idx].to(self.device)
                b_value = rb_data["value"][idx].to(self.device)
                b_rewards = rb_data["reward"][idx].to(self.device)
                b_advantage = advantage[idx]
                b_estim_returns = estim_returns[idx]

                ### Update actor ###
                cur_logprob, entropy = self.actor.eval_action(b_obs, b_action)
                cur_logprob = torch.unsqueeze(cur_logprob, dim=-1)
                mean_entropy = entropy.mean()
                actor_loss = self.actor_loss(cur_logprob, b_logprob, b_advantage, mean_entropy)

                self.actor_opt.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                ### Update critic ###
                cur_value = self.critic(b_obs)
                critic_loss = self.critic_loss(cur_value, b_estim_returns, b_value)

                self.critic_opt.zero_grad()
                critic_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                # Compute average losses
                avg_critic_loss += critic_loss.detach().cpu().numpy().item() / n_minibatches
                avg_actor_loss += actor_loss.detach().cpu().numpy().item() / n_minibatches

            metric_data[Metrics.critic_loss].append(avg_critic_loss)
            metric_data[Metrics.actor_loss].append(avg_actor_loss)

        self.rb.clear()
        return metric_data

    def run_experiment(self):
        self.logger.write({"start": f"{datetime.datetime.now()}"})

        # Get task seed
        env_seed = np.random.randint(low=0, high=int(1e6))

        obs, _ = self.task.reset(seed=env_seed)
        done = False
        st_time = time.time()
        t = 0
        epi_cnt = 0
        while t < self.config.n_steps:
            # Increment timestep
            t += 1

            # Collect the trajectory
            with torch.no_grad():
                obs = torch.tensor(obs).to(torch.float32)
                if obs.ndim <= 1:
                    # Reshape into the batch format
                    obs = obs.unsqueeze(0)

                action, logprob, mean, std = self.actor.act(obs)
                value = self.critic(obs)

            action = action.detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            if len(action.shape) >= 2 and action.shape[0] == 1:
                action = action.squeeze(0)

            # Progress the environment
            next_obs, reward, terminations, truncations, infos = self.task.step(action)
            if np.isnan(next_obs).any() or np.isinf(next_obs).any():
                done = True
                self.rb.buffer["done"][self.rb.idx-1] = True
            else:
                done = terminations or truncations

                # Save to RB
                store_data = {k: v for k, v in zip(self.rb_keys, [obs, logprob, action, value, reward, done])}
                self.rb.register(store_data)

                obs = next_obs

            if t % self.update_freq == 0:
                obs = torch.tensor(obs).to(torch.float32)
                if obs.ndim <= 1:
                    obs = obs.unsqueeze(0)

                # Update parameters and record the metrics
                data = self.update(obs, done, t)
                self.tracker.add(data)

            if done:
                # Summary data for the episode
                episodic_data = {
                    Metrics.episodic_return: [infos["episode"]['r']],
                }

                if epi_cnt % 200 == 0:
                    self.logger.write({"Episode": epi_cnt})
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
                    aggregations = [np.mean, negative_frequency] if "inner_prod" in key else [np.mean]
                    for agg in aggregations:
                        binned_data[f"{key}_{agg.__name__}"] = [np.apply_along_axis(agg, 0, value)]

                # Save the binned data and reset the tracker
                self.binned_statistics.add(binned_data)
                self.tracker.reset(self.tracker.results.keys())

            if t % (self.config.n_steps // 10) == 0:
                # Logging
                hours, mins ,secs = duration(st_time)
                msg = f"Timestep {t} on session {self.session_name}, seed {self.config.seed}: {hours} hrs {mins} mins {secs} sec."
                self.logger.write({f"{t}": msg})

        # Save and record the results
        self.binned_statistics.save_non_empty()
        hours, mins, secs = duration(st_time)
        msg = f"Experiment {self.session_name} ran for {t} steps, {epi_cnt} episodes ({hours} hrs {mins} mins {secs} secs.)."
        print(msg)
        self.logger.write({"summary": msg})
        self.logger.write({"end": f"{datetime.datetime.now()}"})

        self.logger.save()
