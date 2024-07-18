import gym
import numpy as np

class GridEnvWrapper(gym.Env):


    def __init__(self, env, fsa, fsa_init_state="u0"):

        self.env = env 
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.exit_states = self.env.unwrapped.exit_states
        self.PHI_OBJ_TYPES = env.PHI_OBJ_TYPES


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


class FlatQEnvWrapper(gym.Env):

    def __init__(self, env, fsa, fsa_init_state="u0", eval_mode=False):

        self.env = env
        self.fsa = fsa
        self.fsa_init_state = fsa_init_state
        self.exit_states = self.env.unwrapped.exit_states
        self.observation_space = gym.spaces.Box(low=np.zeros(
            3), high=np.ones(3), dtype=np.float32)
        self.action_space = self.env.action_space
        self.w = np.array([1.0,])
        self.initial = []
        for s in self.env.initial:
            self.initial.append(self._merge_states(fsa_init_state, s))
        self.eval_mode = eval_mode


    def get_state(self):

        return self._merge_states(fsa_state=self.fsa_state, state=self.state)

    def reset(self):

        self.fsa_state = self.fsa_init_state
        self.state = tuple(self.env.reset())
        return self._merge_states(fsa_state=self.fsa_state, state=self.state)

    def step(self, action):
        """
            Low-level and high-level transition
        """
        _, _, done, info = self.env.step(action)
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
            info["phi"] = -1000
            return self._merge_states(fsa_state=self.fsa_state, state=state), -1000, True, {'phi': -1000}

        self.fsa_state = next_fsa_state
        if self.eval_mode:
            done = self.fsa.is_terminal(self.fsa_state) or 'TimeLimit.truncated' in info
        else:
            done = self.fsa.is_terminal(self.fsa_state)
        info.pop('TimeLimit.truncated', None)

        # TODO: Add failure case (crash into obstacle)
        info["phi"] = -1
        return self._merge_states(fsa_state=self.fsa_state, state=state), -1, done, info

    def _merge_states(self, fsa_state, state):
        u_index = self.fsa.states.index(fsa_state)
        return (u_index, *state)
