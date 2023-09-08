import numpy as np
import random
import gym
from gym.spaces import Discrete, Box
from abc import ABC, abstractmethod


class GridEnv(ABC, gym.Env):
    metadata = {'render.modes': ['human'],
        'video.frames_per_second': 20}
    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    @property
    def MAP(self):
        raise NotImplementedError

    @property
    def PHI_OBJ_TYPES(self):
        raise NotImplementedError
    
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    References
    ----------
    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start, random_act_prob):
        """
        Creates a new instance of the coffee environment.

        """
        self.random_act_prob = random_act_prob
        self.add_obj_to_start = add_obj_to_start

        self.viewer = None
        self.height, self.width = self.MAP.shape
        self.all_objects = dict(zip(self.PHI_OBJ_TYPES, range(len(self.PHI_OBJ_TYPES))))
        self.initial = []
        self.occupied = set()
        self.object_ids = dict()

        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == '_':
                    self.initial.append((r, c))
                elif self.MAP[r, c] == 'X':
                    self.occupied.add((r, c))
                elif self.MAP[r, c] in self.PHI_OBJ_TYPES:
                    self.object_ids[(r, c)] = len(self.object_ids)
                    if add_obj_to_start:
                        self.initial.append((r, c))

        self.w = np.zeros(self.feat_dim)
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.zeros(
            2), high=np.ones(2))

    def _create_coord_mapping(self):
        """
        Create mapping from coordinates to state id and inverse mapping
        """
        self.state_to_coords = {}
        idx = 0
        for i in range(0, self.MAP.shape[0]):
            for j in range(0, self.MAP.shape[1]):
                if self.MAP[i][j] == "X":
                    continue
                self.state_to_coords[idx] = (i, j)
                idx += 1
        self.coords_to_state = dict(reversed(item) for item in self.state_to_coords.items())

    @abstractmethod
    def _create_transition_function(self):
        raise NotImplementedError

    def _create_transition_function_base(self):
        # Basic movement
        self.P = np.zeros((self.s_dim, self.a_dim, self.s_dim))
        for start_s in range(self.s_dim):
            for eff_a in range(self.a_dim):
                start_coords = self.state_to_coords[start_s]
                new_coords = self.base_movement(start_coords, eff_a)
                new_s = self.coords_to_state[new_coords]
                for a in range(self.a_dim):
                    if eff_a == a:
                        self.P[start_s, a, new_s] += 1-self.random_act_prob
                    else:
                        self.P[start_s, a, new_s] += self.random_act_prob / (self.a_dim-1)
        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)

    @staticmethod
    def state_to_array(state):
        return np.array(state, dtype=np.int32)

    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            self.state = random.choice(self.initial)
        return self.state_to_array(self.state)

    def base_movement(self, coords, action):
        row, col = coords

        if action == self.LEFT:
            col -= 1
        elif action == self.UP:
            row -= 1
        elif action == self.RIGHT:
            col += 1
        elif action == self.DOWN:
            row += 1
        else:
            raise Exception('bad action {}'.format(action))
        if col < 0 or col >= self.width or row < 0 or row >= self.height or (row, col) in self.occupied: # no move
            return coords
        else:
            return (row, col)

    def step(self, action):
        # Movement
        old_state = self.state
        old_state_index = self.coords_to_state[old_state]
        new_state_index = np.random.choice(a=self.s_dim, p=self.P[old_state_index, action])
        new_state = self.state_to_coords[new_state_index]

        self.state = new_state

        # Determine features and rewards
        phi = self.features(old_state, action, new_state)
        reward = np.dot(phi, self.w)
        done = self.is_done(old_state, action, new_state)
        return self.state_to_array(self.state), reward, done, {'phi': phi}

    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        # (y, x), coll = state
        # n_state = self.width + self.height
        # result = np.zeros((n_state + len(coll),))
        # result[y] = 1
        # result[self.height + x] = 1
        # result[n_state:] = np.array(coll)
        # result = result.reshape((1, -1))
        # return result
        raise NotImplementedError()

    def encode_dim(self):
        return self.width + self.height + len(self.object_ids)

    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def is_done(self, state, action, next_state):
        return next_state in self.object_ids

    def features(self, state, action, next_state):
        s1 = next_state
        nc = self.feat_dim
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1.
        return phi

    @property
    def feat_dim(self):
        return len(self.all_objects)

    @property
    def a_dim(self):
        return self.action_space.n

    @property
    def s_dim(self):
        return len(self.state_to_coords)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            max_x, max_y = self.MAP.shape
            square_size = 75

            screen_height = square_size * max_x
            screen_width = square_size * max_y
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.square_map = {}
            for i in range(max_x):
                for j in range(max_y):
                    l = j * square_size
                    r = l + square_size
                    t = max_x * square_size - i * square_size
                    b = t - square_size
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(square)
                    self.viewer.square_map[(i, j)] = square

        for square_coords in self.viewer.square_map:
            square = self.viewer.square_map[square_coords]

            # Agent
            if square_coords == tuple(self.state):
                color = [1, 1, 0]

            # Exit state
            elif square_coords in self.object_ids.keys():
                color = [0, 0, 1]
            elif square_coords in self.occupied:
                color = [0, 0, 0]
            else:
                color = [1, 1, 1]
            square.set_color(*color)
        self.custom_render(square_map=self.viewer.square_map)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def custom_render(self, square_map: dict[tuple[int, int]]):
        pass


class Teleport(GridEnv):
    MAP = np.array([
                 [ ' ', ' ', ' ', ' ', 'TS', ' ', ' ', ' ', ' ', ],
                 [ ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ],
                 [ ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ],
                 [ ' ', ' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', ],
                 [ 'O1', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O2', ],
                 ])
    PHI_OBJ_TYPES = ['O1', 'O2']

    def __init__(self, random_act_prob=0.0):
        """
        Creates a new instance of the coffee environment.

        """
        super().__init__(add_obj_to_start=False, random_act_prob=random_act_prob)
        self.teleport_start = list()
        self.teleport_ends = list()

        # Add teleport
        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == 'TS':
                    self.teleport_start.append((r, c))
                elif self.MAP[r, c] in {'O1', 'O2'}:
                    self.teleport_ends.append((r, c))

        self._create_coord_mapping()
        self._create_transition_function()

    def _create_transition_function(self):
        # Basic grid env transitions
        self. _create_transition_function_base()

        # Specific teleport addition
        teleport_state = self.coords_to_state[self.teleport_start[0]]
        for start_s in range(self.s_dim):
            for a in range(self.a_dim):
                if self.P[start_s, a, teleport_state] >= 0:
                    for i in self.teleport_ends:
                        self.P[start_s, a, self.coords_to_state[i]] += 1.0/len(self.object_ids) * self.P[start_s, a, teleport_state]
                    self.P[start_s, a, teleport_state] = 0

        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)

    def custom_render(self, square_map: dict[tuple[int, int]]):
        for square_coords in square_map:
            square = square_map[square_coords]
            # Teleport
            if square_coords in self.teleport_start:
                color = [0, 1, 1]
            else:
                continue
            square.set_color(*color)


class CoffeeOffice(GridEnv):
    MAP = np.array([[' ', ' ', ' ',   'X', ' ', 'O2', ' ', ' '],
                     [' ', ' ', 'C1', 'X', 'X', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', 'X', 'C2', ' ', ' '],
                     [' ', ' ', ' ', ' ', 'X', ' ', ' ', ' '],
                     [' ', ' ', '_', ' ', ' ', ' ', ' ', ' '],
                     ['O1', ' ', ' ', ' ', ' ', ' ', ' ', ' '], ])
    
    PHI_OBJ_TYPES = ['C1', 'C2', 'O1', 'O2']
    
    
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start=False, random_act_prob=0.0):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob)
        self._create_coord_mapping()
        self._create_transition_function()

    def _create_transition_function(self):
        self._create_transition_function_base()


class OfficeComplex(GridEnv):
    MAP = np.array([[' ', ' ', ' ',   'X', ' ', 'O2', ' ', ' '],
                     [' ', ' ', 'C1', 'X', 'X', ' ', ' ', ' '],
                     [' ', ' ', ' ',  ' ', 'X', 'C2', ' ', ' '],
                     [' ', ' ', ' ',  ' ',  'X', ' ', ' ', ' '],
                     [' ', ' ', ' ',  ' ',  'X', ' ', ' ', ' '],
                     [' ', ' ', ' ',  ' ',  ' ', ' ', 'M', ' '],
                     [' ', ' ', '_',  ' ',  ' ', ' ', ' ', ' '],
                     ['O1', ' ',' ',  ' ',  ' ', ' ', ' ', ' '], ])
    
    PHI_OBJ_TYPES = ['C1', 'C2', 'O1', 'O2', 'M']
    
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    def __init__(self, add_obj_to_start=False, random_act_prob=0.0):
        super().__init__(add_obj_to_start=add_obj_to_start, random_act_prob=random_act_prob)
        self._create_coord_mapping()
        self._create_transition_function()

        exit_states = {}
        for s in self.object_ids:
            symbol = self.MAP[s]
            exit_states[self.PHI_OBJ_TYPES.index(symbol)] = s

        self.exit_states = exit_states


    def _create_transition_function(self):
        self._create_transition_function_base()

class Room(GridEnv):
    MAP = np.array([['X', 'X', 'R', 'X', 'X'],
                 ['X', '_', '_', '_', 'X'],
                 ['P', '_', '_', '_', 'B'],
                 ['X', '_', '_', '_', 'X'],
                 ['X', 'X', 'Y', 'X', 'X']])
    PHI_OBJ_TYPES = ['P', 'R', 'B', 'Y']

    def __init__(self, random_act_prob=0.0):
        super().__init__(add_obj_to_start=False, random_act_prob=random_act_prob)
        self._create_coord_mapping()
        self._create_transition_function()

    def _create_transition_function(self):
        self._create_transition_function_base()


class FourRooms(GridEnv):
    MAP = np.array([
    ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']])
    PHI_OBJ_TYPES = ['1', '2', '3']

    def __init__(self, random_act_prob=0.0):
        super().__init__(add_obj_to_start=False, random_act_prob=random_act_prob)
        # NOTE: Modify this depending on the number of 'shapes' considered (2, 3,).
        self._create_coord_mapping()
        self._create_transition_function()

    def _create_transition_function(self):
        self._create_transition_function_base()




if __name__ == '__main__':
    env = CoffeeOffice(random_act_prob=0.25)
    gamma = 0.99
    w = np.array([1.0, 0.0])

    for i in range(50):
        env.reset()
        while True:
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                break
