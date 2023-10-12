from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch as th
from gym.spaces import Discrete, Box


class RLAlgorithm(ABC):

    def __init__(self, env, device: Union[th.device, str] = 'auto', log_prefix: str = "") -> None:
        self.env = env
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
