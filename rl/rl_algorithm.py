from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch as th
from gym.spaces import Discrete, Box
from fsa.planning import SFFSAValueIteration as VI


class RLAlgorithm(ABC):

    def __init__(self, env, device: Union[th.device, str] = 'auto', fsa_env = None, log_prefix: str = "") -> None:
        self.env = env
        self.fsa_env = fsa_env
        self.observation_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
        self.device = th.device('cuda' if th.cuda.is_available(
        ) else 'cpu') if device == 'auto' else device

        self.num_timesteps = 0
        self.num_episodes = 0
        self.log_prefix = log_prefix

    @abstractmethod
    def eval(self, obs: np.array) -> Union[int, np.array]:
        """Gives the best action for the given observation

        Args:
            obs (np.array): Observation

        Returns:
            np.array or int: Action
        """

    @abstractmethod
    def train(self):
        """Update algorithm's parameters
        """

    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration

        Returns:
            dict: Config
        """

    def q_values(self, obs: np.array) -> np.array:
        """Retrienve q_values for the given observation

        Args:
            obs (np.array): State

        Returns:
            np.array: [q(s,a_1),...,q(s,a_n)]
        """
        raise NotImplementedError

    def evaluate_fsa(self, prints=False):

        # Custom function to evaluate the so-far computed CCS,
        # on a given FSA.

        def evaluate(env, W, num_steps = 100):
    
            env.reset()
            acc_reward = 0

            for _ in range(num_steps):

                (f, state) = env.get_state()
                w = W[f]
               
                action = self.gpi.eval(state, w)        

                if prints: 
                    print(f, w)          
                    print((f, state), np.round(self.gpi.max_q(state, w), 2))
                    print((f, state), 'action', action)

                _, reward, done, _ = env.step(action)
                acc_reward+=reward

                if done:
                    break

            return acc_reward
        

        planning = VI(self.fsa_env, self.gpi, constraint=self.constraint)
        W, _ = planning.traverse(None, k=15)
        if prints:
            print(W)
        acc_reward = evaluate(self.fsa_env, W, num_steps=200)

        return acc_reward