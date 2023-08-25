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

    def state_to_array(self, state):
        return np.array(state, dtype=np.int32)

    def reset(self):
        self.state = random.choice(self.initial)
        return self.state_to_array(self.state)

    def step(self, action):
        old_state = self.state
        row, col = self.state

        # move
        if action == Teleport.LEFT:
            col -= 1
        elif action == Teleport.UP:
            row -= 1
        elif action == Teleport.RIGHT:
            col += 1
        elif action == Teleport.DOWN:
            row += 1
        else:
            raise Exception('bad action {}'.format(action))

        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}

        # into a blocked cell, cannot move
        s1 = (row, col)
        if s1 in self.occupied:
            return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}

        if s1 in self.teleport_start:
            s1 = random.sample(self.teleport_ends, 1)[0]  # Teleport
            self.state = s1
            phi = self.features(old_state, action, self.state)
            reward = np.dot(phi, self.w)
            return self.state_to_array(self.state), reward, True, {'phi': phi}
        # can now move
        # NOTE: The collected part of the state plays a ~similar~ role to a reward machine.
        self.state = s1

        # into a shape cell
        if s1 in self.object_ids:

            self.state = s1
            phi = self.features(old_state, action, self.state)
            reward = np.dot(phi, self.w)
            return self.state_to_array(self.state), reward, True, {'phi': phi}

        # into an empty cell
        return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}

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

    for i in range(20):
        env.reset()
        while True:
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                break