import os
import time
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb as wb
from sfols.rl.rl_algorithm import RLAlgorithm
from sfols.rl.successor_features.gpi import GPI
from sfols.rl.utils.buffer import ReplayBuffer
from sfols.rl.utils.nets import mlp
from sfols.rl.utils.prioritized_buffer import PrioritizedReplayBuffer
from sfols.rl.utils.utils import (eval_mo, huber, layer_init,
                                  linearly_decaying_epsilon, polyak_update)
from torch.utils.tensorboard import SummaryWriter


class Psi(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 phi_dim: int,
                 net_arch: List[int] = [256, 256],
                 normalize_inputs: bool = False,
                 obs_low: np.ndarray = None,
                 obs_high: np.ndarray = None):
        """
        A successor‐feature network.  If `normalize_inputs` is True, we first
        rescale obs into [0,1] via (x - low)/(high - low) before the MLP.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.phi_dim = phi_dim
        self.normalize_inputs = normalize_inputs

        if self.normalize_inputs:
            assert obs_low is not None and obs_high is not None, \
                "When normalize_inputs=True you must pass obs_low and obs_high"
            # register as buffers so they move with .to(device)
            self.register_buffer("obs_low", th.tensor(obs_low, dtype=th.float32))
            self.register_buffer("obs_high", th.tensor(obs_high, dtype=th.float32))

        # the rest is unchanged
        self.net = mlp(self.obs_dim, self.action_dim * self.phi_dim, net_arch)
        self.apply(layer_init)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [batch, obs_dim]
        if self.normalize_inputs:
            # safe divide
            denom = (self.obs_high - self.obs_low).clamp(min=1e-6)
            x = (x - self.obs_low) / denom
        q = self.net(x)  # [batch, action_dim*phi_dim]
        return q.view(-1, self.action_dim, self.phi_dim)


class SFDQN(RLAlgorithm):

    def __init__(self,
                 env,
                 fsa_env=None,
                 gpi: GPI = None,
                 learning_rate: float = 3e-4,
                 initial_epsilon: float = 0.01,
                 final_epsilon: float = 0.01,
                 epsilon_decay_steps: int = None,  # None == fixed epsilon
                 tau: float = 1.0,
                 target_net_update_freq: int = 1000,  # ignored if tau != 1.0
                 buffer_size: int = int(1e6),
                 net_arch: List = [256, 256],
                 model_arch: List = [200, 200, 200, 200],
                 batch_size: int = 256,
                 learning_starts: int = 100,
                 gradient_updates: int = 1,
                 gamma: float = 0.99,
                 per: bool = False,
                 min_priority: float = 1.0,
                 project_name: str = 'sfqqn',
                 experiment_name: str = 'sfdqn',
                 log: bool = True,
                 log_prefix: str = "",
                 device: Union[th.device, str] = 'auto',
                 normalize_inputs=False):

        super(SFDQN, self).__init__(env, device, fsa_env=fsa_env)
        self.gpi = gpi
        self.phi_dim = len(self.env.w)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.per = per
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.model_arch = model_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates

        obs_low, obs_high = env.get_observation_bounds()

        self.psi_net = Psi(self.observation_dim, self.action_dim, self.phi_dim, net_arch=net_arch,
                           normalize_inputs=normalize_inputs, obs_low=obs_low, obs_high=obs_high).to(self.device)
        self.target_psi_net = Psi(self.observation_dim, self.action_dim, self.phi_dim, net_arch=net_arch,
                                  normalize_inputs=normalize_inputs, obs_low=obs_low, obs_high=obs_high).to(self.device)
        self.target_psi_net.load_state_dict(self.psi_net.state_dict())
        for param in self.target_psi_net.parameters():
            param.requires_grad = False
        self.psi_optim = optim.Adam(self.psi_net.parameters(), lr=self.learning_rate)

        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(self.observation_dim, 1, rew_dim=self.phi_dim,
                                                         max_size=buffer_size, action_dtype=np.uint8)
        else:
            self.replay_buffer = ReplayBuffer(self.observation_dim, 1, rew_dim=self.phi_dim, max_size=buffer_size,
                                              action_dtype=np.uint8)
        self.min_priority = min_priority
        self.alpha = 0.6

        self.log = log
        if log:
            pass

    def define_wandb_metrics(self):
        # episode‑level (logged when an episode ends):
        wb.define_metric(f"{self.log_prefix}episode")
        wb.define_metric(f"{self.log_prefix}episode_reward",
                         step_metric=f"{self.log_prefix}episode")

        # timestep‑level (logged every gradient update / eval step):
        wb.define_metric("learning/timestep")
        wb.define_metric(f"{self.log_prefix}critic_loss",
                         step_metric="learning/timestep")
        wb.define_metric(f"{self.log_prefix}epsilon",
                         step_metric="learning/timestep")

        # if you use PER, track priority:
        if self.per:
            wb.define_metric(f"{self.log_prefix}mean_priority",
                             step_metric="learning/timestep")

        # evaluation metrics
        wb.define_metric(f"{self.log_prefix}eval/total_reward",
                         step_metric="learning/timestep")
        wb.define_metric(f"{self.log_prefix}eval/discounted_return",
                         step_metric="learning/timestep")

    def get_config(self):
        return {'env_id': self.env.unwrapped.spec.id,
                'learning_rate': self.learning_rate,
                'initial_epsilon': self.initial_epsilon,
                'epsilon_decay_steps:': self.epsilon_decay_steps,
                'batch_size': self.batch_size,
                'tau': self.tau,
                'gamma': self.gamma,
                'net_arch': self.net_arch,
                'model_arch': self.model_arch,
                'gradient_updates': self.gradient_updates,
                'buffer_size': self.buffer_size,
                'learning_starts': self.learning_starts}

    def save(self, save_replay_buffer=True, save_dir='weights/'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params['psi_net_state_dict'] = self.psi_net.state_dict()
        saved_params['target_psi_net_state_dict'] = self.target_psi_net.state_dict()
        saved_params['psi_nets_optimizer_state_dict'] = self.psi_optim.state_dict()
        if save_replay_buffer:
            saved_params['replay_buffer'] = self.replay_buffer
        th.save(saved_params, save_dir + "/" + self.experiment_name + '.tar')

    def load(self, path, load_replay_buffer=True):
        params = th.load(path)
        self.psi_net.load_state_dict(params['psi_net_state_dict'])
        self.target_psi_net.load_state_dict(params['target_psi_net_state_dict'])
        self.psi_optim.load_state_dict(params['psi_nets_optimizer_state_dict'])
        if load_replay_buffer and 'replay_buffer' in params:
            self.replay_buffer = params['replay_buffer']

    def sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def train(self, w: th.tensor):
        for _ in range(self.gradient_updates):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self.sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.sample_batch_experiences()

            with th.no_grad():
                if self.gpi is not None:
                    psi_values = th.stack([policy.psi_net(s_next_obs) for policy in self.gpi.policies])
                    psa = th.einsum('r,psar->psa', w, psi_values)
                    sa, ac = th.max(psa, dim=2)
                    polices = th.argmax(sa, dim=0)
                    max_acts = ac.gather(0, polices.reshape(1, -1)).squeeze(0)
                    psi_targets = self.target_psi_net(s_next_obs)
                    psi_targets = psi_targets.gather(1, max_acts.long().reshape(-1, 1, 1).expand(psi_targets.size(0), 1,
                                                                                                 psi_targets.size(2)))
                else:
                    psi_values = th.einsum('r,sar->sa', w, self.psi_net(s_next_obs))
                    max_acts = th.argmax(psi_values, dim=1)
                    psi_targets = self.target_psi_net(s_next_obs)
                    psi_targets = psi_targets.gather(1, max_acts.long().reshape(-1, 1, 1).expand(psi_targets.size(0), 1,
                                                                                                 psi_targets.size(2)))

                target_psi = psi_targets.reshape(-1, self.phi_dim)
                target_psi = (s_rewards + (1 - s_dones) * self.gamma * target_psi).detach()

            psi_value = self.psi_net(s_obs)
            psi_value = psi_value.gather(1, s_actions.long().reshape(-1, 1, 1).expand(psi_value.size(0), 1,
                                                                                      psi_value.size(2)))
            psi_value = psi_value.reshape(-1, self.phi_dim)
            td_error = (psi_value - target_psi)
            critic_loss = huber(td_error.abs(), min_priority=self.min_priority)

            self.psi_optim.zero_grad()
            critic_loss.backward()
            #th.nn.utils.clip_grad_norm_(self.psi_net.parameters(), 50.0)
            self.psi_optim.step()

            if self.per:
                td_error = td_error[:len(idxes)].detach()
                per = th.einsum('r,sr->s', w, td_error).abs()
                priority = per.clamp(min=self.min_priority).pow(self.alpha).cpu().numpy().flatten()
                self.replay_buffer.update_priorities(idxes, priority)

        if self.tau != 1.0 or self.num_timesteps % self.target_net_update_freq == 0:
            polyak_update(self.psi_net.parameters(), self.target_psi_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps,
                                                     self.learning_starts, self.final_epsilon)

        if self.log and self.num_timesteps % 100 == 0:
            if self.per:
                wb.log({"metrics/mean_priority": np.mean(priority)}, step=self.num_timesteps)
                wb.log({"metrics/mean_td_error_w": per.abs().mean().item()}, step=self.num_timesteps)
            wb.log({"losses/critic_loss": critic_loss.item()}, step=self.num_timesteps)
            wb.log({"metrics/epsilon": self.epsilon}, step=self.num_timesteps)

        if not self.police_indices:
            return
        this_policy_ind = self.police_indices[-1]
        if self.gpi is not None and this_policy_ind != len(self.gpi.policies) - 1:
            this_task = th.tensor(self.gpi.tasks[this_policy_ind]).float().to(self.device)
            this_policy = self.gpi.policies[this_policy_ind]
            this_policy.num_timesteps += 1
            with th.no_grad():
                psa = th.einsum('r,psar->psa', this_task, psi_values)
                sa, ac = th.max(psa, dim=2)
                polices = th.argmax(sa, dim=0)
                max_acts = ac.gather(0, polices.reshape(1, -1)).squeeze(0)
                psi_targets = this_policy.target_psi_net(s_next_obs)
                psi_targets = psi_targets.gather(1, max_acts.long().reshape(-1, 1, 1).expand(psi_targets.size(0), 1,
                                                                                             psi_targets.size(2)))

                target_psi = psi_targets.reshape(-1, self.phi_dim)
                target_psi = (s_rewards + (1 - s_dones) * self.gamma * target_psi).detach()

            psi_value = this_policy.psi_net(s_obs)
            psi_value = psi_value.gather(1, s_actions.long().reshape(-1, 1, 1).expand(psi_value.size(0), 1,
                                                                                      psi_value.size(2)))
            psi_value = psi_value.reshape(-1, self.phi_dim)
            td_error = (psi_value - target_psi)
            critic_loss = huber(td_error.abs(), min_priority=self.min_priority)

            this_policy.psi_optim.zero_grad()
            critic_loss.backward()
            #th.nn.utils.clip_grad_norm_(self.psi_net.parameters(), 50.0)
            this_policy.psi_optim.step()

            if this_policy.tau != 1.0 or this_policy.num_timesteps % this_policy.target_net_update_freq == 0:
                polyak_update(this_policy.psi_net.parameters(), this_policy.target_psi_net.parameters(),
                              this_policy.tau)

    def q_values(self, obs: th.tensor, w: th.tensor) -> th.tensor:
        with th.no_grad():
            psi_values = self.psi_net(obs)
            q = th.einsum('r,sar->sa', w, psi_values)
            return q

    def eval(self, obs: np.array, w: np.array) -> int:
        obs = th.tensor(obs).float().to(self.device)
        w = th.tensor(w).float().to(self.device)
        if self.gpi is not None:
            return self.gpi.eval(obs, w)
        else:
            return th.argmax(self.q_values(obs, w), dim=1).item()

    def act(self, obs: th.tensor, w: th.tensor) -> np.array:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.gpi is not None:
                action, policy_index = self.gpi.eval(obs, w, return_policy_index=True)
                self.police_indices.append(policy_index)
                return action
            else:
                return th.argmax(self.q_values(obs, w), dim=1).item()

    def learn(self, total_timesteps, fsa_env=None, total_episodes=None, reset_num_timesteps=True, eval_env=None,
              eval_freq=1000, w=np.array([1.0, 0.0]),
              M=[np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]):
        episode_reward = 0.0
        episode_vec_reward = np.zeros_like(w)
        num_episodes = 0
        self.police_indices = []
        obs, done = self.env.reset(), False

        self.env.unwrapped.w = w
        tensor_w = th.tensor(w).float().to(self.device)

        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        for _ in range(1, total_timesteps + 1):
            if total_episodes is not None and num_episodes == total_episodes:
                break

            self.num_timesteps += 1

            if self.num_timesteps < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.tensor(obs).float().to(self.device), tensor_w)

            next_obs, reward, done, info = self.env.step(action)

            terminal = done if 'TimeLimit.truncated' not in info else not info['TimeLimit.truncated']
            self.replay_buffer.add(obs, action, info['phi'], next_obs, terminal)

            if self.num_timesteps >= self.learning_starts:
                self.train(tensor_w)

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                total_reward, discounted_return, total_vec_r, total_vec_return = eval_mo(self, eval_env, w)
                wb.log({"eval/total_reward": total_reward}, step=self.num_timesteps)
                wb.log({"eval/discounted_return": discounted_return}, step=self.num_timesteps)
                for i in range(episode_vec_reward.shape[0]):
                    wb.log({f"eval/total_reward_obj{i}": total_vec_r[i]}, step=self.num_timesteps)
                    wb.log({f"eval/return_obj{i}": total_vec_return[i]}, step=self.num_timesteps)

            episode_reward += reward
            episode_vec_reward += info['phi']
            if done:
                obs, done = self.env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 10 == 0:
                    print(
                        f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}")
                if self.log:
                    wb.log({'metrics/policy_index': np.array(self.police_indices), 'global_step': self.num_timesteps})
                    self.police_indices = []
                    wb.log({"metrics/episode": self.num_episodes}, step=self.num_timesteps)
                    wb.log({"metrics/episode_reward": episode_reward}, step=self.num_timesteps)
                    for i in range(episode_vec_reward.shape[0]):
                        wb.log({f"metrics/episode_reward_obj{i}": episode_vec_reward[i]}, step=self.num_timesteps)

                episode_reward = 0.0
                episode_vec_reward = np.zeros_like(w)
            else:
                obs = next_obs
