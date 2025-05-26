import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union, List, Tuple
import wandb

from sfols.plotting.plotting import plot_q_vals
from sfols.rl.rl_algorithm import RLAlgorithm
from sfols.rl.utils.buffer import ReplayBuffer
from sfols.rl.utils.prioritized_buffer import PrioritizedReplayBuffer
from sfols.rl.utils.utils import linearly_decaying_epsilon, polyak_update, huber
from sfols.rl.utils.nets import mlp


class QNet(nn.Module):
    """
    Q-network with optional normalization of continuous inputs.
    Splits the input vector into:
      - cont: first `cont_dim` dims, normalized via (x - low)/ (high - low)
      - fsa: remaining `fsa_dim` dims (one-hot), left unchanged
    Then passes concatenated tensor through an MLP to output `action_dim` values.
    """

    def __init__(self,
                 cont_dim: int,
                 fsa_dim: int,
                 action_dim: int,
                 net_arch: List[int],
                 normalize_inputs: bool = False,
                 obs_low: np.ndarray = None,
                 obs_high: np.ndarray = None):
        super().__init__()
        self.cont_dim = cont_dim
        self.fsa_dim = fsa_dim
        self.action_dim = action_dim
        self.normalize_inputs = normalize_inputs
        input_dim = cont_dim + fsa_dim

        if self.normalize_inputs:
            assert obs_low is not None and obs_high is not None, \
                "Must provide obs_low and obs_high when normalize_inputs=True"
            self.register_buffer("obs_low", th.tensor(obs_low, dtype=th.float32))
            self.register_buffer("obs_high", th.tensor(obs_high, dtype=th.float32))

        self.net = mlp(input_dim, action_dim, net_arch)
        # optional initialization (if desired)
        # self.apply(layer_init)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [batch, cont_dim + fsa_dim]
        if self.normalize_inputs:
            cont = x[..., :self.cont_dim]
            denom = (self.obs_high - self.obs_low).clamp(min=1e-6)
            cont = (cont - self.obs_low) / denom
            fsa = x[..., self.cont_dim:]
            x = th.cat([cont, fsa], dim=-1)
        return self.net(x)  # [batch, action_dim]


class DQN(RLAlgorithm):
    """
    Deep Q-Network for continuous GridEnv with an FSA state.

    Observation: tuple (fsa_state_idx:int, continuous_xy: np.ndarray)
    Input to QNet:  [x,y] normalized + one-hot FSA vector
    Output: Q(s,a) for discrete actions.
    """

    def __init__(self,
                 env,
                 eval_env,
                 n_fsa_states: int,
                 net_arch: List[int] = [256, 256],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 initial_epsilon: float = 1.0,
                 final_epsilon: float = 0.1,
                 epsilon_decay_steps: Optional[int] = 10000,
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 learning_starts: int = 1000,
                 target_update_freq: int = 1000,
                 tau: float = 1.0,
                 per: bool = False,
                 min_priority: float = 1.0,
                 normalize_inputs: bool = True,
                 log: bool = True,
                 log_prefix: str = "",
                 device: Union[str, th.device] = 'auto',
                 **kwargs) -> None:
        super().__init__(env, device, fsa_env=None, log_prefix=log_prefix)
        self.eval_env = eval_env
        self.n_fsa_states = n_fsa_states
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.batch_size = batch_size
        self.per = per
        self.min_priority = min_priority

        # continuous coordinate bounds
        obs_low, obs_high = env.env.get_observation_bounds()
        # Build Q-net and target Q-net
        self.q_net = QNet(
            cont_dim=obs_low.shape[0],
            fsa_dim=n_fsa_states,
            action_dim=env.action_space.n,
            net_arch=net_arch,
            normalize_inputs=normalize_inputs,
            obs_low=obs_low,
            obs_high=obs_high
        ).to(self.device)
        self.target_q_net = QNet(
            cont_dim=obs_low.shape[0],
            fsa_dim=n_fsa_states,
            action_dim=env.action_space.n,
            net_arch=net_arch,
            normalize_inputs=normalize_inputs,
            obs_low=obs_low,
            obs_high=obs_high
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_q_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # replay buffer
        if per:
            self.replay_buffer = PrioritizedReplayBuffer(
                obs_dim=obs_low.shape[0] + n_fsa_states,
                action_dim=1,
                rew_dim=1,
                max_size=buffer_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                obs_dim=obs_low.shape[0] + n_fsa_states,
                action_dim=1,
                rew_dim=1,
                max_size=buffer_size
            )

        # logging
        self.log = log
        if log:
            wandb.define_metric(f"{log_prefix}epsilon", step_metric="learning/timestep")
            wandb.define_metric(f"{log_prefix}critic_loss", step_metric="learning/timestep")
            wandb.define_metric(f"eval/reward", step_metric="learning/timestep")

    def _build_input(self, fsa_idx: int, cont: np.ndarray) -> th.Tensor:
        """
        Create the combined input: normalized cont + one-hot fsa
        """
        # one-hot
        fsa_onehot = np.zeros(self.n_fsa_states, dtype=np.float32)
        fsa_onehot[fsa_idx] = 1.0
        arr = np.concatenate([cont.astype(np.float32), fsa_onehot])
        return th.tensor(arr, dtype=th.float32, device=self.device)

    def act(self, state: Tuple[int, np.ndarray]) -> int:
        fsa_idx, cont = state
        # random exploration
        if self.num_timesteps < self.learning_starts or np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        inp = self._build_input(fsa_idx, cont).unsqueeze(0)
        with th.no_grad():
            q = self.q_net(inp)
        return int(q.argmax(dim=1).item())

    def q_values(self, state: Tuple[int, np.ndarray]) -> np.ndarray:
        fsa_idx, cont = state
        inp = self._build_input(fsa_idx, cont).unsqueeze(0)
        with th.no_grad():
            q = self.q_net(inp)
        return q.cpu().numpy().squeeze(0)

    def sample_batch(self):
        return self.replay_buffer.sample(
            batch_size=self.batch_size,
            to_tensor=True,
            device=self.device
        )

    def learn(self,
              total_timesteps: int,
              total_episodes: Optional[int] = None,
              reset_num_timesteps: bool = False,
              eval_freq: int = 1000,
              **kwargs) -> None:
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_episodes = 0
            self.epsilon = self.initial_epsilon

        state = self.env.reset()
        done = False
        ep_ret = 0.0

        for t in range(1, total_timesteps + 1):
            self.num_timesteps += 1
            fsa_idx, cont = state[0], np.array((state[1], state[2]))
            a = self.act((fsa_idx, cont))
            next_state, reward, done, _ = self.env.step(a)
            if reward != -1:
                print(reward)
            next_idx, next_cont = next_state[0], np.array((next_state[1], next_state[2]))

            # store to buffer
            inp = np.concatenate([cont.astype(np.float32),
                                  np.eye(self.n_fsa_states)[fsa_idx]])
            inp_next = np.concatenate([next_cont.astype(np.float32),
                                       np.eye(self.n_fsa_states)[next_idx]])
            self.replay_buffer.add(inp, a, reward, inp_next, done)

            # update
            if self.num_timesteps >= self.learning_starts:
                obs_b, act_b, rew_b, nxt_b, done_b = self.sample_batch()
                q_vals = self.q_net(obs_b).gather(1, act_b.long()).squeeze(1)
                with th.no_grad():
                    next_q = self.target_q_net(nxt_b).max(1)[0]
                    target = rew_b + self.gamma * (1 - done_b) * next_q
                td = q_vals - target
                loss = huber(td)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                # target update
                if t % self.target_update_freq == 0 or self.tau < 1.0:
                    polyak_update(
                        self.q_net.parameters(),
                        self.target_q_net.parameters(),
                        self.tau
                    )
                # epsilon decay
                if self.epsilon_decay_steps:
                    self.epsilon = linearly_decaying_epsilon(
                        self.initial_epsilon,
                        self.epsilon_decay_steps,
                        self.num_timesteps,
                        self.learning_starts,
                        self.final_epsilon
                    )
                # logging
                if self.log and t % eval_freq == 0:
                    fsa_reward = self.evaluate()
                    wandb.log({
                        f"{self.log_prefix}epsilon": self.epsilon,
                        f"{self.log_prefix}critic_loss": loss.mean().item(),
                        "learning/timestep": self.num_timesteps,
                        "learning/fsa_reward": fsa_reward
                    })

            # episode bookkeeping
            ep_ret += reward
            state = next_state
            if done:
                self.num_episodes += 1
                if self.log:
                    wandb.log({
                        f"{self.log_prefix}episode_return": ep_ret,
                        f"{self.log_prefix}episode": self.num_episodes,
                        "learning/timestep": self.num_timesteps
                    })
                state = self.env.reset()
                done = False
                ep_ret = 0.0

    def train(self, *args, **kwargs):
        """
        Stub to satisfy the abstract RLAlgorithm interface.
        All of our Q-learning actually runs inside `learn()`, so we don’t
        need this, but we must define it.
        """
        return None

    def get_config(self) -> dict:
        """
        Return a serializable dict of hyperparameters for logging or checkpointing.
        """
        return {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "target_update_freq": self.target_update_freq,
            "tau": self.tau,
            "per": self.per,
            "net_arch": self.q_net.net.hidden_layers if hasattr(self.q_net, "net") else None,
            "normalize_inputs": self.q_net.normalize_inputs,
        }

    def evaluate(self, num_steps: int = 200) -> float:
        state = self.eval_env.reset(use_low_level_init_state=True, use_fsa_init_state=True)
        acc = 0.0
        done = False
        for _ in range(num_steps):
            fsa_idx, cont = state[0], np.array((state[1], state[2]))
            a = self.eval((fsa_idx, cont))
            state, r, done, _ = self.eval_env.step(a)
            acc += r
            if done:
                break
        return acc

    def eval(self, state: Tuple[int, np.ndarray]) -> int:
        fsa_idx, cont = state
        inp = self._build_input(fsa_idx, cont).unsqueeze(0)
        with th.no_grad():
            q = self.q_net(inp)
        return int(q.argmax(dim=1).item())

    def save(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"dqn_policy.pt")
        th.save({
            'q_state': self.q_net.state_dict(),
            'target_q_state': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load(cls, env, eval_env, n_fsa_states: int, path: str, **init_kwargs):
        # re-instantiate
        agent = cls(env, eval_env, n_fsa_states, **init_kwargs)

        # pick device
        device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

        # load everything onto that device
        data = th.load(os.path.join(path, "dqn_policy.pt"), map_location=device)

        # restore and move nets
        agent.q_net.load_state_dict(data['q_state'])
        agent.q_net.to(device)

        agent.target_q_net.load_state_dict(data['target_q_state'])
        agent.target_q_net.to(device)

        # optimizer state is already on CPU; 
        # if you want its tensors on GPU you’ll need to loop through its state:
        agent.optimizer.load_state_dict(data['optimizer'])
        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.to(device)

        return agent

    def get_arrow_data(self, uidx: int, batch_size: int = 256):
        """
        Returns the quiver‐plot params (X,Y,U,V,C) for the greedy policy
        in FSA‐state `uidx` over all continuous centers.
        """
        # 1) collect all cell‐centers
        centers = self.env.env.get_all_valid_continuous_states_centers()  # list of (y,x)
        N = len(centers)

        # 2) build a tensor input for each center
        #    _build_input returns a torch.Tensor of shape [cont_dim+fsa_dim]
        inputs = [ self._build_input(uidx, np.array(center))
                   for center in centers ]

        all_actions = []
        all_qvals   = []

        # 3) batch through q_net
        for start in range(0, N, batch_size):
            batch = inputs[start : start + batch_size]  # list of Tensors
            batch_tensor = th.stack(batch, dim=0)        # [B, input_dim]
            with th.no_grad():
                q_out = self.q_net(batch_tensor)        # [B, action_dim]
            # pick greedy actions & values
            acts = q_out.argmax(dim=1).cpu().numpy()   # shape [B,]
            qmax = q_out.max(dim=1).values.cpu().numpy()
            all_actions.append(acts)
            all_qvals.append(qmax)

        # 4) concatenate back to full length
        actions = np.concatenate(all_actions, axis=0)  # [N,]
        qvals   = np.concatenate(all_qvals,   axis=0)  # [N,]

        # 5) hand off to your env’s quiver‐builder
        return self.env.env.get_arrow_data(actions, qvals, states=centers)

    def plot_q_vals(self, activation_data=None, base_dir=None, show=True):
        def _plot_one(uidx):
            save_path = f"{base_dir}/qvals_{uidx}.png" if base_dir is not None else None
            arrow_data = self.get_arrow_data(uidx)
            plot_q_vals(self.env.env, arrow_data=arrow_data, activation_data=activation_data,
                        save_path=save_path, show=show, goal_prop=f"u{uidx}")

        for i in range(self.n_fsa_states - 1):
            _plot_one(i)
