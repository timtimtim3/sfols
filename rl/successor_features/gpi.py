import json
import re
import time
import os
import glob
import pickle as pkl
from fsa.planning import get_augmented_phi
from sfols.plotting.plotting import plot_q_vals
from sfols.rl.rl_algorithm import RLAlgorithm
from typing import Union, Callable, Optional, List, Tuple, Any
import numpy as np
import torch as th
from copy import deepcopy
from typing import Union, Tuple

EvalReturnType = Union[
    int,  # Only the action is returned
    Tuple[int, int],  # Action and policy index
    Tuple[int, float],  # Action and q-value
    Tuple[int, int, float]  # Action, policy index, and q-value
]


class GPI(RLAlgorithm):

    def __init__(self,
                 env,
                 algorithm_constructor: Callable,
                 fsa_env=None,
                 log: bool = True,
                 device: Union[th.device, str] = 'auto',
                 planning_constraint: Optional[dict] = None,
                 psis_are_augmented=False,
                 ValueIteration=None):

        super(GPI, self).__init__(env, device, fsa_env=fsa_env, planning_constraint=planning_constraint)

        self.algorithm_constructor = algorithm_constructor
        self.policies = []
        self.tasks = []
        self.learned_policies = 0
        self.num_timesteps = 0

        self.log = log
        self.psis_are_augmented = psis_are_augmented

        self.ValueIteration = ValueIteration

    def eval(self, obs, w, return_policy_index=False, exclude=None, return_q_val=False) -> EvalReturnType:
        """

            This takes in an observation and a weight vector and returns an action.
            What this actually returns in the GPI policy given the current CCS.

        """
        if not hasattr(self.policies[0], 'q_table'):
            if isinstance(obs, np.ndarray):
                obs = th.tensor(obs).float().to(self.device)
                w = th.tensor(w).float().to(self.device)
            q_vals = th.stack([policy.q_values(obs, w)
                               for policy in self.policies])
            max_q, a = th.max(q_vals, dim=2)
            policy_index = th.argmax(max_q)

            if return_policy_index:
                return a[policy_index].detach().long().item(), policy_index.item()
            return a[policy_index].detach().long().item()
        else:
            q_vals = np.stack([policy.q_values(obs, w)
                               for policy in self.policies if policy is not exclude])
            policy_index, action = np.unravel_index(
                np.random.choice(np.flatnonzero(q_vals == q_vals.max())), q_vals.shape
            )
            selected_qval = q_vals[policy_index, action]

            if return_policy_index and return_q_val:
                return action, policy_index, selected_qval
            elif return_policy_index:
                return action, policy_index
            elif return_q_val:
                return action, selected_qval
            return action

    def eval_planning(self, obs, w, return_policy_index=False, exclude=None, return_q_val=False, uidx=None) -> EvalReturnType:
        """

            This takes in an observation and a weight vector and returns an action.
            What this actually returns in the GPI policy given the current CCS.

        """
        if self.psis_are_augmented:
            all_policy_augmented_psis = np.stack([policy.get_augmented_psis(uidx, obs) for policy in self.policies])

            q_vals = np.stack([augmented_psis @ w for augmented_psis in all_policy_augmented_psis])
            policy_index, action = np.unravel_index(
                np.random.choice(np.flatnonzero(q_vals == q_vals.max())), q_vals.shape
            )
            selected_qval = q_vals[policy_index, action]

            if return_policy_index and return_q_val:
                return action, policy_index, selected_qval
            elif return_policy_index:
                return action, policy_index
            elif return_q_val:
                return action, selected_qval
            return action
        else:
            return self.eval(obs, w, return_policy_index=return_policy_index, exclude=exclude,
                             return_q_val=return_q_val)

    def get_gpi_policy_on_w(self, w, uidx=None):
        actions, policy_indices, qvals = {}, {}, {}
        for state in self.policies[0].q_table.keys():
            action, policy_index, selected_qval = self.eval_planning(state, w, return_policy_index=True, return_q_val=True,
                                                            uidx=uidx)
            actions[state] = action
            policy_indices[state] = policy_index
            qvals[state] = selected_qval
        return actions, policy_indices, qvals

    # Given obs returns
    def max_q(self, obs, w, tensor=False, exclude=None, return_np=False):
        if tensor:
            if not isinstance(obs, th.Tensor):
                obs = th.tensor(obs, dtype=th.float32, device=self.device)
            if not isinstance(w, th.Tensor):
                w = th.tensor(w, dtype=th.float32, device=self.device)

            with th.no_grad():
                psi_values = th.stack([policy.target_psi_net(
                    obs) for policy in self.policies if policy is not exclude])
                q_values = th.einsum('r,psar->psa', w, psi_values)
                max_q, a = th.max(q_values, dim=2)
                polices = th.argmax(max_q, dim=0)
                max_acts = a.gather(0, polices.unsqueeze(0)).squeeze(0)
                psi_i = psi_values.gather(0, polices.reshape(
                    1, -1, 1, 1).expand(1, psi_values.size(1), psi_values.size(2), psi_values.size(3))).squeeze(0)
                max_psis = psi_i.gather(
                    1, max_acts.reshape(-1, 1, 1).expand(psi_i.size(0), 1, psi_i.size(2))).squeeze(1)
                if max_psis.dim() == 2 and max_psis.size(0) == 1:
                    max_psis = max_psis.squeeze(0)  # now [phi_dim]
                if return_np:
                    return max_psis.detach().cpu().numpy()
                return max_psis
        else:
            # q_vals is now a matrix of (n_policies, action_dim) where each row is the Q-values for that policy for the
            # different actions
            q_vals = np.stack([policy.q_values(obs, w)
                               for policy in self.policies])
            # finds the index of the maximum Q-value in the entire q_vals matrix, then uses unravel_index to convert
            # the flat index into a (row, column) index, so we get the best policy and corresponding best action index
            policy_ind, action = np.unravel_index(
                np.argmax(q_vals), q_vals.shape)
            return self.policies[policy_ind].q_table[tuple(obs)][action]  # returns the sf of the max Q

    def delete_policies(self, delete_indx):
        for i in sorted(delete_indx, reverse=True):
            self.policies.pop(i)
            self.tasks.pop(i)

    def learn(self,
              w,
              total_timesteps,
              total_episodes=None,
              reset_num_timesteps=False,
              eval_freq=1000,
              use_gpi=True,
              reset_learning_starts=True,
              new_policy=True,
              reuse_value_ind=None,
              **kwargs
              ):

        # Creates new policy
        if new_policy:
            new_policy = self.algorithm_constructor(log_prefix=f"policies/policy{self.learned_policies}/")
            self.policies.append(new_policy)

        # Adds new w
        self.tasks.append(w)

        # Sets gpi reference of a new policy to self
        self.policies[-1].gpi = self if use_gpi else None

        # Not important - logging
        if self.log:
            self.policies[-1].log = self.log

        if len(self.policies) > 1:
            # Copy steps and episodes for further counting?
            self.policies[-1].num_timesteps = self.policies[-2].num_timesteps
            self.policies[-1].num_episodes = self.policies[-2].num_episodes
            if reset_learning_starts:
                # to reset exploration schedule
                self.policies[-1].learning_starts = self.policies[-2].num_timesteps

        # If set to an index copies the q function from previous policy as initialization
        if reuse_value_ind is not None:
            if hasattr(self.policies[-1], 'q_table'):
                self.policies[-1].q_table = deepcopy(
                    self.policies[reuse_value_ind].q_table)
            else:
                self.policies[-1].psi_net.load_state_dict(
                    self.policies[reuse_value_ind].psi_net.state_dict())
                self.policies[-1].target_psi_net.load_state_dict(
                    self.policies[reuse_value_ind].psi_net.state_dict())

            # Copy replay buffer
            self.policies[-1].replay_buffer = self.policies[-2].replay_buffer

        # New policy learns using new w
        self.policies[-1].learn(w=w,
                                total_timesteps=total_timesteps,
                                total_episodes=total_episodes,
                                reset_num_timesteps=reset_num_timesteps,
                                eval_freq=eval_freq,
                                **kwargs)

        self.learned_policies += 1

    @property
    def gamma(self):
        return self.policies[0].gamma

    def train(self):
        pass

    def get_config(self) -> dict:
        if len(self.policies) > 0:
            return self.policies[0].get_config()
        return {}

    def evaluate_fsa(self, fsa_env, ValueIteration=None) -> int:

        # Custom function to evaluate the so-far computed CCS,
        # on a given FSA.
        if ValueIteration is not None:
            pass
        elif self.ValueIteration is not None:
            ValueIteration = self.ValueIteration
        else:
            from fsa.planning import SFFSAValueIteration as ValueIteration

        planning = ValueIteration(fsa_env, self, constraint=self.planning_constraint)
        W, _ = planning.traverse(None, num_iters=15)

        acc_reward = GPI.evaluate(self, fsa_env, W, num_steps=200)

        return acc_reward

    @staticmethod
    def evaluate(gpi,
                 env,
                 W: dict,
                 num_steps: Optional[int] = 200,
                 render=False,
                 initial_sleep=3,
                 sleep_time=0.3) -> int:

        env.reset()
        acc_reward = 0

        if render:
            env.env.render()
            time.sleep(initial_sleep)  # Add delay for better visualization

        for _ in range(num_steps):

            (f, state) = env.get_state()
            if gpi.psis_are_augmented:
                w = np.asarray(list(W.values())).reshape(-1)
            else:
                w = W[f]

            action = gpi.eval_planning(state, w, uidx=int(f.split('u')[1]))

            _, reward, done, _ = env.step(action)
            acc_reward += reward

            if render:
                env.env.render()
                time.sleep(sleep_time)  # Add delay for better visualization

            if done:
                break

        return acc_reward

    def do_rollout(self,
                   gpi,
                   env,
                   W: dict,
                   n_fsa_states,
                   feat_dim,
                   num_steps: Optional[int] = 200,
                   render=False,
                   initial_sleep=3,
                   sleep_time=0.3,
                   gamma=0.99) -> tuple[list[Any], list[Any], list[Any], list[Any]]:

        env.reset()
        acc_reward = 0
        gamma_t = 1.0
        all_gamma_t = []
        all_max_q = []
        all_v = []

        if render:
            env.env.render()
            time.sleep(initial_sleep)  # Add delay for better visualization

        for _ in range(num_steps):

            (f, state) = env.get_state()
            if self.psis_are_augmented:
                w = np.asarray(list(W.values())).reshape(-1)
            else:
                w = W[f]

            uidx = int(f.split('u')[1])
            action, q_val = gpi.eval_planning(state, w, uidx=uidx, return_q_val=True)

            _, reward, done, _ = env.step(action)
            acc_reward += reward
            gamma_t *= gamma

            phi = self.env.env.features(state=None, action=None, next_state=state)
            augmented_phi = get_augmented_phi(phi, uidx, n_fsa_states, feat_dim)

            v_val = augmented_phi @ w

            all_gamma_t.append(gamma_t)
            all_max_q.append(q_val)
            all_v.append(v_val)

            if render:
                env.env.render()
                time.sleep(sleep_time)  # Add delay for better visualization

            if done:
                break

        all_gamma_t_v_values = all_gamma_t
        all_gamma_t_q_values = [1.0] + all_gamma_t[:-1]

        return all_max_q, all_v, list(reversed(all_gamma_t_v_values)), list(reversed(all_gamma_t_q_values))

    def evaluate_single_policy(self, policy_index: int, env, task_index: int = None, num_steps: Optional[int] = 200,
                               render: bool = False, verbose: bool = False, initial_sleep: float = 3,
                               get_stuck_max: int = 10) -> int:
        """
        Evaluates a single policy (identified by policy_index) on the environment.

        Parameters:
          - policy_index (int): The index of the policy in self.policies to use for action selection.
          - env: The environment (NOT wrapped with the FSA) on which to evaluate.
          - task_index (int): Which task (weight vector) to eval on
          - num_steps (Optional[int]): Maximum number of steps for evaluation.
          - render (bool): Whether to render the environment at each step.
          - verbose (bool): Whether to print.
          - initial_sleep (float): How many seconds to sleep before the agent starts.
          - get_stuck_max (int): If the agent gets stuck in the same state for get_stuck_max times we quit.

        Returns:
          - acc_reward (int): The accumulated reward from the evaluation.
        """
        task_idx = task_index if task_index is not None else policy_index
        w = self.tasks[task_idx]

        if verbose:
            print(
                f"Evaluating policy: {policy_index}, on task: {task_idx}, with w: {np.array2string(w, precision=2, separator=', ')}.")

        env.reset()
        acc_reward = 0
        selected_policy = self.policies[policy_index]

        # Optional: Render an initial frame if requested
        if render:
            env.env.render()
            time.sleep(initial_sleep)

        stuck_count = 0
        for _ in range(num_steps):
            # Get the current combined state: (fsa_state, env_state)
            state = self.env.state
            action = selected_policy.eval(state, w)

            # Step in the environment using the chosen action
            new_state, reward, done, _ = env.step(action)
            acc_reward += reward

            if np.array_equal(new_state, state):
                stuck_count += 1

            if render:
                env.env.render()
                time.sleep(0.3)

            if done or get_stuck_max == stuck_count:
                break

        return acc_reward

    def evaluate_all_single_policies(self, env, num_steps: Optional[int] = 200,
                                     render: bool = False, verbose: bool = False,
                                     get_stuck_max: int = 10) -> list[int]:
        """
        Evaluates all policies on the environment.

        Parameters:
          - env: The environment (NOT wrapped with the FSA) on which to evaluate.
          - num_steps (Optional[int]): Maximum number of steps for evaluation.
          - render (bool): Whether to render the environment at each step.
          - verbose (bool): Whether to print.
          - get_stuck_max (int): If the agent gets stuck in the same state for get_stuck_max times we quit.

        Returns:
          - acc_reward (list[int]): The accumulated reward from the evaluation.
        """
        acc_rewards = []
        initial_sleep = 3
        for idx in range(len(self.policies)):
            acc_reward = self.evaluate_single_policy(idx, env, render=render, verbose=verbose,
                                                     get_stuck_max=get_stuck_max,
                                                     initial_sleep=initial_sleep, num_steps=num_steps)
            acc_rewards.append(acc_reward)
            initial_sleep = 1
        return acc_rewards

    def load_tasks(self, task_dir):
        # try pickle first
        pkl_path = os.path.join(task_dir, "tasks.pkl")
        json_path = os.path.join(task_dir, "tasks.json")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                tasks_data = pkl.load(f)
            self.tasks = tasks_data
            print(f"Loaded {len(self.tasks)} tasks from {pkl_path} (pickle)")

        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                tasks_list = json.load(f)
            # rebuild numpy arrays
            self.tasks = [np.array(item) for item in tasks_list]
            print(f"Loaded {len(self.tasks)} tasks from {json_path} (JSON)")

        else:
            self.tasks = []
            print(f"No tasks.pkl or tasks.json found in {task_dir}. " 
                  "self.tasks is set to an empty list.")

    def load_policies(self, policy_dir):
        # collect (idx, filename) for any q‑table or DQN checkpoint
        file_infos = []
        for fname in os.listdir(policy_dir):
            m_json = re.match(r"qtable_pol(\d+)\.json$", fname)
            m_pt = re.match(r"dqn(\d+)\.(?:pt|pth)$", fname)  # or .pt / .pth
            if m_json:
                idx = int(m_json.group(1))
                file_infos.append((idx, fname))
            elif m_pt:
                idx = int(m_pt.group(1))
                file_infos.append((idx, fname))

        # now sort by index and load
        for idx, fname in sorted(file_infos, key=lambda x: x[0]):
            path = os.path.join(policy_dir, fname)
            policy = self.algorithm_constructor(log_prefix=f"load-policy-{idx}")
            self.policies.append(policy)
            policy.load(path)

    # def load_policies(self, policy_dir: str, q_tables):
    #     """
    #     Loads saved policies and tasks from the specified directory and assigns them to self.policies and self.tasks.
    #
    #     Parameters:
    #         policy_dir (str): The directory where the policy pickle files and tasks.pkl are stored.
    #         q_tables (dict): Dictionary holding policy indicices mapped to q-table dictionaries.
    #     """
    #
    #     # Load policy pickle files from the directory
    #     pkl_files = sorted(glob.glob(os.path.join(policy_dir, "discovered_policy_*.pkl")))
    #     print(f"Loading {len(pkl_files)} policies from {policy_dir}")
    #     for i, pkl_path in enumerate(pkl_files):
    #         with open(pkl_path, "rb") as fp:
    #             policy_data = pkl.load(fp)
    #         # Reconstruct a new policy by calling the algorithm constructor.
    #         # This ensures that the policy object has the correct class and methods.
    #         policy = self.algorithm_constructor(log_prefix="load-policy")
    #         # Restore attributes from the unpickled dictionary
    #         for k, v in policy_data.items():
    #             if k == 'q_table':
    #                 continue
    #             setattr(policy, k, v)
    #
    #         policy.q_table = q_tables[i]
    #         # Insert the policy into the GPI agent's policy list
    #         self.policies.append(policy)
    #     print(f"Loaded {len(self.policies)} policies into GPI agent.")

    def save_policies(self, base_dir):
        for i, policy in enumerate(self.policies):
            policy.save(base_dir, policy_idx=i)

    def save_tasks(self, base_dir, as_json=False, as_pickle=True):
        os.makedirs(base_dir, exist_ok=True)

        if as_json:
            # Convert each numpy array to a regular list
            tasks_list = [arr.tolist() for arr in self.tasks]
            json_path = os.path.join(base_dir, "tasks.json")
            with open(json_path, "w") as fp:
                json.dump(tasks_list, fp, indent=2)
            print(f"Saved {len(self.tasks)} tasks to {json_path} (JSON)")

        if as_pickle:
            tasks = self.tasks
            tasks_path = os.path.join(base_dir, "tasks.pkl")
            with open(tasks_path, "wb") as fp:
                pkl.dump(tasks, fp)
            print(f"Saved {len(tasks)} tasks to {tasks_path} (pickle)")

    def plot_q_vals(self, activation_data, base_dir=None, unique_symbol_for_centers=False, show=True, policy_id=None):
        if policy_id is not None:
            policy = self.policies[policy_id]
            w = self.tasks[policy_id]
            save_path = f"{base_dir}/qvals_pol{policy_id}.png" if base_dir is not None else None
            arrow_data = policy.get_arrow_data(w)
            plot_q_vals(w, self.env, arrow_data=arrow_data, activation_data=activation_data,
                        save_path=save_path, show=show, unique_symbol_for_centers=unique_symbol_for_centers)
        else:
            for i, (policy, w) in enumerate(zip(self.policies, self.tasks)):
                save_path = f"{base_dir}/qvals_pol{i}.png" if base_dir is not None else None
                arrow_data = policy.get_arrow_data(w)
                plot_q_vals(w, self.env, arrow_data=arrow_data, activation_data=activation_data,
                            save_path=save_path, show=show, unique_symbol_for_centers=unique_symbol_for_centers)
