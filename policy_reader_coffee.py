import pickle as pkl
import numpy as np
import os
# import argparse


actions = ["LEFT", "UP", "RIGHT", "DOWN"]

# Enumerate the relevant locations (check the environment)
coffee1 = (2, 3, 0, 0)
coffee2 = (3, 6, 0, 0)
init_state = (5, 3, 0, 0)


def _get_successor_features(dirpath):

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)

    return SFs


def coffee_then_office_deterministic():

    # 1) Get the SF representstion for the policies in the CCS for the office task.

    SFs = _get_successor_features(os.path.abspath("policies/Office-v0"))

    # 2) We set the weights to 1 for each of the offices as any of the sastisfies the task,
    # we compute the q-function for each policy and for each 'coffee' location and
    # then we can apply GPI (see Theorem 1 in [1])

    weights_office = np.asarray([1, 1])

    q_values_coffee1 = np.asarray(
        [np.dot(q[coffee1], weights_office) for q in SFs])
    q_values_coffee2 = np.asarray(
        [np.dot(q[coffee2], weights_office) for q in SFs])

    value_coffee1 = np.max(q_values_coffee1)
    value_coffee2 = np.max(q_values_coffee2)
    print(f'Value for coffee at {coffee1}', value_coffee1)
    print(f'Value for coffee at {coffee2}', value_coffee2)

    # Such values there after are the weights to combine the coffee subtask.
    weights_coffee = np.asarray([value_coffee1, value_coffee2])

    SFs = _get_successor_features(os.path.abspath("policies/Coffee-v0"))

    q_values_init_state = np.asarray(
        [np.dot(q[init_state], weights_coffee) for q in SFs])

    argmax = np.unravel_index(
        np.argmax(q_values_init_state), q_values_init_state.shape)

    value_init_state = np.max(q_values_init_state)
    print(f'Value for init_state at {init_state}', value_init_state)
    print("Optimal action at init_state after composing:", actions[argmax[-1]])

    # The value for going to the RIGHT is 0.7737809040955714 which is exactly .95 ** (N-1) for N = 6 that is the length of the
    # optimal path.


def coffee_then_office_stochastic():

    # This is the same version, but has a 0.3 of random noise probability in the effect of the action.
    # This means, that with 0.3 probability, the action could fail in any of the 3 other actions.
    # 1) Get the SF representstion for the policies in the CCS for the office task.

    SFs = _get_successor_features(os.path.abspath("policies/Office-v1"))

    # 2) We set the weights to 1 for each of the offices as any of the sastisfies the task,
    # we compute the q-function for each policy and for each 'coffee' location and
    # then we can apply GPI (see Theorem 1 in [1])

    weights_office = np.asarray([1, 1])

    q_values_coffee1 = np.asarray(
        [np.dot(q[coffee1], weights_office) for q in SFs])
    q_values_coffee2 = np.asarray(
        [np.dot(q[coffee2], weights_office) for q in SFs])

    value_coffee1 = np.max(q_values_coffee1)
    action_coffee1 = np.unravel_index(
        np.argmax(q_values_coffee1), q_values_coffee1.shape)[-1]

    value_coffee2 = np.max(q_values_coffee2)
    action_coffee2 = np.unravel_index(
        np.argmax(q_values_coffee2), q_values_coffee2.shape)[-1]
    print(f'Value for coffee at {coffee1}',
          value_coffee1, actions[action_coffee1])
    print(f'Value for coffee at {coffee2}',
          value_coffee2, actions[action_coffee2])

    # PATH; (5, 3, 0, 0) [RIGHT] --> (5, 4, 0, 0)  [RIGHT] --> (5, 5, 0, 0)  [RIGHT] --
    # --> (5, 6, 0, 0)  [UP]   --> (4, 6, 0, 0)  [UP]  -->  (3, 6, 0, 0)

    # Such values there after are the weights to combine the coffee subtask.
    weights_coffee = np.asarray([value_coffee1, value_coffee2])

    SFs = _get_successor_features(os.path.abspath("policies/Coffee-v1"))

    q_values_init_state = np.asarray(
        [np.dot(q[(5, 3, 0, 0)], weights_coffee) for q in SFs])

    argmax = np.unravel_index(
        np.argmax(q_values_init_state), q_values_init_state.shape)

    value_init_state = np.max(q_values_init_state)
    print(f'Value for init_state at {init_state}', value_init_state)
    print("Optimal action at init_state after composing:", actions[argmax[-1]])
    print(q_values_init_state)

    # The value for going to the RIGHT is 0.7737809040955714 which is exactly .95 ** (N-1) for N = 6 that is the length of the
    # optimal path.


if __name__ == "__main__":

    coffee_then_office_stochastic()


"""
References
----------
[1] Barreto, André et al. "Successor Features for Transfer in Reinforcement Learning".
"""
