import numpy as np
import random
import gym
from gym.spaces import Discrete, Box


class Teleport(gym.Env):
    MAP = np.array([['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
                 ['X', ' ', ' ', ' ', ' ', 'TS', ' ', ' ', ' ', ' ', 'X'],
                 ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
                 ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
                 ['X', ' ', ' ', ' ', ' ', '_', ' ', ' ', ' ', ' ', 'X'],
                 ['X', 'O1', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O2', 'X'],
                 ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']])
    metadata = {'render.modes': ['human'],
        'video.frames_per_second': 20}
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    References
    ----------
    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    def __init__(self):
        """
        Creates a new instance of the coffee environment.

        """
        self.viewer = None

        self.height, self.width = self.MAP.shape
        # self.shape_rewards = shape_rewards
        # sorted(list(shape_rewards.keys()))
        object_types = ['O1', 'O2']
        self.all_objects = dict(zip(object_types, range(len(object_types))))

        self.goal = None
        self.initial = []
        self.occupied = set()
        self.object_ids = dict()
        self.teleport_start = list()
        self.teleport_ends = list()
        for c in range(self.width):
            for r in range(self.height):
                if self.MAP[r, c] == '_':
                    self.initial.append((r, c))
                elif self.MAP[r, c] == 'TS':
                    self.teleport_start.append((r, c))
                elif self.MAP[r, c] == 'X':
                    self.occupied.add((r, c))
                elif self.MAP[r, c] in {'O1', 'O2'}:
                    self.teleport_ends.append((r, c))
                    self.object_ids[(r, c)] = len(self.object_ids)

        # NOTE: Modify this depending on the number of 'objects' considered.
        # Here 2 coffe machine and 1 office locations are considered
        self.w = np.zeros(2)
        self.action_space = Discrete(4)
        # NOTE: The osbservation_space is (X, Y, <objects_collected>)
        self.observation_space = Box(low=np.zeros(
            2), high=np.ones(2))
        self._create_coord_mapping()
        self._create_transition_function()

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

    def _create_transition_function(self):
        # Basic movement
        s_dim = len(self.state_to_coords)
        a_dim = self.action_space.n
        self.P = np.zeros((s_dim, a_dim, s_dim))
        for start_s in range(s_dim):
            for a in range(a_dim):
                start_coords = self.state_to_coords[start_s]
                new_coords = self.base_movement(start_coords, a)
                new_s = self.coords_to_state[new_coords]
                self.P[start_s, a, new_s] = 1

        # Specific additions
        teleport_state = self.coords_to_state[(1, 5)]
        for start_s in range(s_dim):
            for a in range(a_dim):
                if self.P[start_s, a, teleport_state] == 1:
                    self.P[start_s, a, teleport_state] = 0
                    for i in self.teleport_ends:
                        self.P[start_s, a, self.coords_to_state[i]] = 1.0/len(self.object_ids)


        # sanity check
        assert np.allclose(np.sum(self.P, axis=2), 1)


    def state_to_array(self, state):
        return np.array(state, dtype=np.int32)

    def reset(self):
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


    def get_q_sf(self, w: np.array, gamma: float):
        s_dim = len(self.state_to_coords)
        a_dim = self.action_space.n
        phi_dim = 2
        Q_sf = np.zeros(shape=(s_dim, a_dim, phi_dim), dtype=np.float32)
        while True:
            Q_new = np.zeros_like(Q_sf)
            for s_old in range(s_dim):
                for a in range(a_dim):
                    q = 0
                    for s_new in range(s_dim):
                        prob = self.P[s_old, a, s_new]
                        if not prob:
                            continue
                        features = self.features(self.state_to_coords[s_old], a, self.state_to_coords[s_new])
                        done = self.is_done(self.state_to_coords[s_old], a, self.state_to_coords[s_new])
                        a_new = np.argmax(Q_sf[s_new] @ w)
                        q += prob * (features + gamma * (1-done) * Q_sf[s_new, a_new])
                    Q_new[s_old, a] = q

            if np.allclose(Q_sf, Q_new):
                break
            else:
                Q_sf = Q_new
        # Probably need to refactor this at some point?
        return Q_sf

    def step(self, action):
        old_state = self.state

        # Base movement
        new_state = self.base_movement(self.state, action)

        # Teleport check
        if new_state in self.teleport_start:
            new_state = random.sample(self.teleport_ends, 1)[0]  # Teleport
            self.state = new_state

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
        (y, x), coll = state
        n_state = self.width + self.height
        result = np.zeros((n_state + len(coll),))
        result[y] = 1
        result[self.height + x] = 1
        result[n_state:] = np.array(coll)
        result = result.reshape((1, -1))
        return result

    def encode_dim(self):
        return self.width + self.height + len(self.object_ids)

    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def is_done(self, state, action, next_state):
        return next_state in self.object_ids

    def features(self, state, action, next_state):
        s1 = next_state
        nc = len(self.all_objects)
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            y, x = s1
            object_index = self.all_objects[self.MAP[y, x]]
            phi[object_index] = 1.
        elif s1 == self.goal:
            phi[nc] = np.ones(nc, dtype=np.float32)

        return phi

    def feature_dim(self):
        return len(self.all_objects)

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

            # Teleport
            elif square_coords in self.teleport_start:
                color = [0, 1, 1]
            elif square_coords in self.teleport_ends:
                color = [1, 0, 1]
            elif square_coords in self.occupied:
                color = [0, 0, 0]
            else:
                color = [1, 1, 1]
            square.set_color(*color)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = Teleport()
    gamma = 0.99
    w = np.array([1.0, 0.0])
    q_sf = env.get_q_sf(w=w, gamma=gamma)

    for i in range(20):
        env.reset()
        while True:
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                break