# -*- coding: UTF-8 -*-
# Code from: https://github.com/mike-gimelfarb/deep-successor-features-for-transfer/blob/main/source/tasks/gridworld.py
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box

MAZE = np.array([['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
                 ['X', ' ', ' ', ' ', 'X', ' ', 'O2', ' ', ' ', 'X'],
                 ['X', ' ', ' ', 'C1', 'X', 'X', ' ', ' ', ' ', 'X'],
                 ['X', ' ', ' ', ' ', ' ', 'X', 'C2', ' ', ' ', 'X'],
                 ['X', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'X'],
                 ['X', ' ', ' ', '_', ' ', ' ', ' ', ' ', ' ', 'X'],
                 ['X', 'O1', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
                 ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']])


class Coffee(gym.Env):
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

        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                G indicates a goal state (terminal state)
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
                # {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                elif maze[r, c] in {'C1', 'C2', 'O1', 'O2'}:
                    self.object_ids[(r, c)] = len(self.object_ids)

        # NOTE: Modify this depending on the number of 'objects' considered.
        # Here 2 coffe machine and 1 office locations are considered
        self.w = np.zeros(4)
        self.action_space = Discrete(4)
        # NOTE: The osbservation_space is (X, Y, <objects_collected>)
        self.observation_space = Box(low=np.zeros(
            2 + len(self.object_ids)), high=np.ones(2 + len(self.object_ids)))

    def state_to_array(self, state):
        s = [element for tupl in state for element in tupl]
        return np.array(s, dtype=np.int32)

    def reset(self):
        self.state = (random.choice(self.initial), tuple(
            0 for _ in range(len(self.object_ids))))
        return self.state_to_array(self.state)

    def step(self, action):
        old_state = self.state
        (row, col), collected = self.state

        # move
        if action == Coffee.LEFT:
            col -= 1
        elif action == Coffee.UP:
            row -= 1
        elif action == Coffee.RIGHT:
            col += 1
        elif action == Coffee.DOWN:
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

        # can now move
        # NOTE: The collected part of the state plays a ~similar~ role to a reward machine.
        self.state = (s1, collected)

        # into a goal cell
        if s1 == self.goal:
            # NOTE: Goal state giving a reward of 1? Does it get back-propagated?
            phi = np.ones(len(self.all_objects), dtype=np.float32)
            return self.state_to_array(self.state), 1., True, {'phi': phi}

        # into a shape cell
        if s1 in self.object_ids:

            object_id = self.object_ids[s1]
            if collected[object_id] == 1:
                # already collected this flag
                return self.state_to_array(self.state), 0., False, {'phi': np.zeros(len(self.all_objects), dtype=np.float32)}
            else:
                # collect the new flag
                collected = list(collected)
                collected[object_id] = 1
                collected = tuple(collected)
                self.state = (s1, collected)
                phi = self.features(old_state, action, self.state)
                reward = np.dot(phi, self.w)
                return self.state_to_array(self.state), reward, False, {'phi': phi}

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
        s1, _ = next_state
        _, collected = state
        nc = len(self.all_objects)
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.object_ids:
            if collected[self.object_ids[s1]] != 1:
                y, x = s1
                object_index = self.all_objects[self.maze[y, x]]
                phi[object_index] = 1.
        elif s1 == self.goal:
            phi[nc] = np.ones(nc, dtype=np.float32)

        return phi

    def feature_dim(self):
        return len(self.all_objects)
