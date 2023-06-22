from typing import Any
from gym.core import Env
import networkx as nx
import gym


class CoffeeRewardMachine(gym.Wrapper):

    """
        This is a Reward Machine wrapper for the Coffee environment. The idea is to implement the task specification as a RW that wraps the 
        underlying environment. This component is resposible for both keeping track of the status of the track and outputting the scalar rewards.
        #NOTE: Actually caveat about the <reward> 
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.graph = nx.DiGraph(("u0", "u1", "T"))

    def reset(self) -> Any:
        self.current_node = "u0"
        return super().reset()

    def step(self, action) -> Any:
        obs, _, terminated, info = self.env.step(action)

        if (self.current_node == "u0") and (1 in info["phi"][:2]):

            pass

        elif if
