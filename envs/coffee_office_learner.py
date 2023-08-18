# -*- coding: UTF-8 -*-
# Code from: https://github.com/mike-gimelfarb/deep-successor-features-for-transfer/blob/main/source/tasks/gridworld.py
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box

MAZE = np.array([[' ', ' ', ' ', 'X', ' ', 'O2', ' ', ' '],
                 [' ', ' ', 'C1', 'X', 'X', ' ', ' ', ' '],
                 [' ', ' ', ' ', ' ', 'X', 'C2', ' ', ' '],
                 [' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ' ],
                 [' ', ' ', '_', ' ', ' ', ' ', ' ', ' ' ],
                 ['O1', ' ', ' ', ' ', ' ', ' ', ' ', ' '],])


class CoffeeOffice(gym.Env):
    """
    A simplified version of the office environment introduced in [1].
    This simplified version consists of 2 coffee machines and 2 office locations.

    References
    ----------
    [1] Icarte, RT, et al. "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning".
    """

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    def __init__(self, maze=MAZE):
        """
        Creates a new instance of the coffee environment.

        This environment is to be wrapped by a Reward Machine (RM) since the reward function is clearly non-markovian.

        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                _ indicates an initial state (there can be multiple, and one is selected at random
                    at the start of each episode)
                X indicates a barrier
                Ci, Oi indicates the type of cell (either coffee machine or office location)
                entries containing other characters are treated as regular empty cells
        object_rewards : dict
            a dictionary mapping the type of object (C1, O1, ... ) to a corresponding reward to provide
            to the agent for collecting an object of that type
            # TODO: What exactly is this line above?
        """
        self.height, self.width = maze.shape
        self.maze = maze
        # self.shape_rewards = shape_rewards
        # sorted(list(shape_rewards.keys()))
        object_types = ['C1', 'C2', 'O1', 'O2']
        self.all_objects = dict(zip(object_types, range(len(object_types))))

        self.goal = None
        self.initial = []
        self.occupied = set()
        self.object_ids = dict()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == 'G':
                    self.goal = (r, c)
                elif maze[r, c] == '_':
                    self.initial.append((r, c))
                elif maze[r, c] == 'X':
                    self.occupied.add((r, c))
                # The objects ids are as follow -> C1:0, C2:1, O1:2, O2:3
                elif maze[r, c] in ['C1', 'C2', 'O1', 'O2']:
                    self.object_ids[(r, c)] = self.all_objects[maze[r, c]]

        # NOTE: Modify this depending on the number of 'objects' considered.
        # Here 2 coffe machine and 1 office locations are considered
        self.w = np.zeros(4)
        self.action_space = Discrete(4)
        # NOTE: The state is a location tuple (X, Y)
        self.observation_space = Box(low=np.zeros(2), high=np.ones(2))

        # Collected.
        self.collected = 4*[0]

    def state_to_array(self, state):
        return np.array(state, dtype=np.int32)

    def reset(self, state=None):

        if state is not None:
            self.state = np.asarray(state)
        else:
            self.state = random.choice(self.initial)

        return self.state_to_array(self.state)

    def step(self, action):
        old_state = self.state

        row, col = self.state

        # move
        if action == CoffeeOffice.LEFT:
            col -= 1
        elif action == CoffeeOffice.UP:
            row -= 1
        elif action == CoffeeOffice.RIGHT:
            col += 1
        elif action == CoffeeOffice.DOWN:
            row += 1
        else:
            raise Exception('bad action {}'.format(action))

        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}

        # into a blocked cell, cannot move
        next_state = (row, col)
        if next_state in self.occupied:
            return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}

        # Legal move within limits, change env state
        self.state = next_state

        # into a goal cell
        if next_state == self.goal:
            # NOTE: Goal state giving a reward of 1? Does it get back-propagated?
            phi = np.ones(len(self.all_objects), dtype=np.float32)
            return self.state_to_array(self.state), 1., True, {'phi': phi}

        # into a shape cell
        if next_state in self.object_ids:

            # collect the new flag
            self.state = next_state 
            phi = self.features(old_state, action, self.state)
            reward = np.dot(phi, self.w)

            return self.state_to_array(self.state), reward, True, {'phi': phi}

        # into an empty cell
        return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}

    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING                                         ==
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
    # SUCCESSOR FEATURES                                                       ==
    # ===========================================================================
    def features(self, state, action, next_state):

        collected = self.collected

        nc = len(self.all_objects)
        phi = np.zeros(nc, dtype=np.float32)
        if next_state in self.object_ids:
            y, x = next_state
            object_index = self.all_objects[self.maze[y, x]]
            if collected[object_index]:
                return phi
            phi[object_index] = 1.
            collected[object_index] = 1
            return phi
        elif next_state == self.goal:
            phi[nc] = np.ones(nc, dtype=np.float32)
            return phi

    def feature_dim(self):
        return len(self.all_objects)
