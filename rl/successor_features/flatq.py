import json
from typing import Optional
import numpy as np
import wandb
from sfols.plotting.plotting import plot_q_vals
from sfols.rl.rl_algorithm import RLAlgorithm


class FlatQ(RLAlgorithm):
    def __init__(self,
                 env,
                 eval_env,
                 n_fsa_states,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 initial_epsilon: float = 1.0,
                 final_epsilon: float = 0.1,
                 epsilon_decay_steps: int = 10000,
                 log: bool = True,
                 log_prefix: str = "",
                 **kwargs):
        super().__init__(env, device=None, fsa_env=None, log_prefix=log_prefix)
        self.eval_env = eval_env
        self.n_fsa_states = n_fsa_states
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        # our plain‐vanilla Q-table
        self.q_table = {}  # maps state‐tuple → np.zeros(n_actions,)

        # logging
        self.log = log
        if self.log:
            wandb.define_metric(f"{log_prefix}epsilon", step_metric="learning/timestep")
            wandb.define_metric(f"{log_prefix}max_td_error", step_metric="learning/timestep")
            wandb.define_metric(f"{log_prefix}episode_return", step_metric=f"{log_prefix}episode")

    def _get_q(self, state):
        """Return the Q-vector for this state, creating if needed."""
        st = tuple(state)
        if st not in self.q_table:
            self.q_table[st] = np.zeros(self.action_dim, dtype=np.float32)
        return self.q_table[st]

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self._get_q(obs)

    def train(self, s, a, r, s_next, done):
        """Do one Q-update and return the scalar TD error."""
        q_s      = self._get_q(s)
        q_s_next = self._get_q(s_next)
        target   = r + (0.0 if done else self.gamma * q_s_next.max())
        td_error = target - q_s[a]
        q_s[a]  += self.alpha * td_error
        return abs(td_error)

    def learn(self,
              total_timesteps: int,
              total_episodes: Optional[int] = None,
              reset_num_timesteps: bool = False,
              eval_freq: int = 1000):
        """
        Very similar shape to your old `learn(...)`:
        - epsilon‐greedy action selection
        - step through env
        - Q-update via train()
        - logging every eval_freq timesteps
        - reset on done, track episode returns
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_episodes  = 0

        obs    = self.env.reset()
        done   = False
        ep_ret = 0.0
        run_avg_td = 0.0

        for t in range(1, total_timesteps + 1):
            if total_episodes is not None and self.num_episodes >= total_episodes:
                break

            self.num_timesteps += 1

            # 1) ε-greedy action
            if np.random.rand() < self.epsilon:
                a = self.env.action_space.sample()
            else:
                a = int(np.argmax(self._get_q(obs)))

            # 2) step
            next_obs, reward, done, info = self.env.step(a)

            if reward != -1:
                print(reward)

            # 3) Q-update
            td_err = self.train(obs, a, reward, next_obs, done)
            run_avg_td += (td_err - run_avg_td) * 0.01

            # 4) bookkeeping
            ep_ret += reward
            obs = next_obs

            # 5) decay ε
            if self.epsilon_decay_steps:
                frac = min(1.0, self.num_timesteps / self.epsilon_decay_steps)
                self.epsilon = self.initial_epsilon + frac * (self.final_epsilon - self.initial_epsilon)

            # 6) periodic logging
            if t % eval_freq == 0 and self.log:
                fsa_reward = self.evaluate()

                wandb.log({
                    f"{self.log_prefix}epsilon":       self.epsilon,
                    f"{self.log_prefix}max_td_error":  run_avg_td,
                    "learning/timestep":               self.num_timesteps,
                    "learning/fsa_reward": fsa_reward
                })

            # 7) end of episode
            if done:
                self.num_episodes += 1
                if self.log:
                    wandb.log({
                        f"{self.log_prefix}episode":        self.num_episodes,
                        f"{self.log_prefix}episode_return": ep_ret,
                    })
                obs, ep_ret, done = self.env.reset(), 0.0, False

        return self.q_table  # return the learned table

    def evaluate(self, num_steps=200):
        state = self.eval_env.reset(use_low_level_init_state=True, use_fsa_init_state=True)

        acc_reward = 0
        for _ in range(num_steps):
            action = self.eval(state)

            state, reward, done, _ = self.eval_env.step(action)
            acc_reward += reward

            if done:
                break
        return acc_reward

    def eval(self, obs: np.array) -> int:
        """Greedy wrt Q, no exploration."""
        return int(np.argmax(self._get_q(obs)))

    def get_config(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
        }

    def save(self, base_dir: str):
        # Serialize q_table to JSON
        serializable = {
            str(k): v.tolist()
            for k, v in self.q_table.items()
        }
        with open(f"{base_dir}/q_table.json", "w") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, env, path: str, **init_kwargs):
        # Reconstruct the agent and then load its q_table
        agent = cls(env, **init_kwargs)
        with open(f"{path}/q_table.json", "r") as f:
            data = json.load(f)
        agent.q_table = {
            eval(k): np.array(v, dtype=np.float32)
            for k, v in data.items()
        }
        return agent

    def get_arrow_data(self, uidx):
        actions, max_qvals, states = [], [], []

        for state, qvals in self.q_table.items():
            fsa_state = state[0]
            if fsa_state != uidx:
                continue
            coords = (state[1], state[2])
            max_qval = np.max(qvals)
            max_action = np.argmax(qvals)
            max_qvals.append(max_qval)
            actions.append(max_action)
            states.append(coords)

        return self.env.env.get_arrow_data(np.array(actions), np.array(max_qvals), states)

    def plot_q_vals(self, activation_data=None, base_dir=None, show=True):
        def _plot_one(uidx):
            save_path = f"{base_dir}/qvals_{uidx}.png" if base_dir is not None else None
            arrow_data = self.get_arrow_data(uidx)
            plot_q_vals(self.env.env, arrow_data=arrow_data, activation_data=activation_data,
                        save_path=save_path, show=show)

        for i in range(self.n_fsa_states - 1):
            _plot_one(i)
