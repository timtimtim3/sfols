import gym
import numpy as np

class GridEnvWrapper(gym.Env):


    def __init__(self, env, fsa, fsa_init_state="u0"):

        self.env = env 
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.exit_states = self.env.unwrapped.exit_states

    def get_state(self):

        return (self.fsa_state, tuple(self.env.state))

    def reset(self):

        self.fsa_state = self.fsa_init_state
        self.state = tuple(self.env.reset())
        
        return (self.fsa_state, self.state)

    def step(self, action):
        """
            Low-level and high-level transition
        """
        _, _, _, info = self.env.step(action) 
        prop = info["proposition"]
        state = self.env.state
        f_state = self.fsa_state

        neighbors = self.fsa.get_neighbors(self.fsa_state)
        satisfied = [prop in self.fsa.get_predicate((f_state, n)) for n in neighbors]

        next_fsa_state = None

        if any(satisfied):
            next_fsa_state = neighbors[satisfied.index(True)]

        if next_fsa_state is None:
            next_fsa_state = f_state

        if self.env.MAP[state] == "O":
            return (self.fsa_state, state), -1000, True, {"proposition" : prop}

        self.fsa_state = next_fsa_state

        done = self.fsa.is_terminal(self.fsa_state)

        # TODO: Add failure case (crash into obstacle)

        return (self.fsa_state, state), -1, done, {"proposition" : prop}
