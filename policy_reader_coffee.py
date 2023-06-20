import pickle as pkl
import numpy as np
import os
# import argparse

if __name__ == "__main__":

    actions = ["LEFT", "UP", "RIGHT", "DOWN"]

    # Enumerate the relevant locations (check the environment)
    coffee1 = (2, 3, 0, 0)
    coffee2 = (3, 6, 0, 0)
    init_state = (5, 3, 0, 0)

    # 1) Get the SF representstion for the policies in the CCS for the office task.
    dirpath = os.path.abspath("policies/Office-v0")

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)

    # We set the weights to 1 for each of the offices as any of the sastisfies the task.
    w = np.asarray([1, 1])

    # We compute the q-function for each policy and for each 'coffee' location.
    # Then we can apply GPI (see Theorem 1 in [1])

    q_values_coffee1 = np.asarray([np.dot(q[coffee1], w) for q in SFs])
    q_values_coffee2 = np.asarray([np.dot(q[coffee2], w) for q in SFs])

    value_coffee1 = np.max(q_values_coffee1)
    value_coffee2 = np.max(q_values_coffee2)
    print(f'Value for coffee at {coffee1}', value_coffee1)
    print(f'Value for coffee at {coffee2}', value_coffee2)

    # Such values there after are the weights to combine the coffee subtask.
    w = np.asarray([value_coffee1, value_coffee2])

    # Now get the policies in the CCS in the coffee subtask
    dirpath = os.path.abspath("policies/Coffee-v0")

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)

    # We compute the q-function for each policy for the initial state.
    # Then we can apply GPI (see Theorem 1 in [1])

    q_values_init_state = np.asarray([np.dot(q[init_state], w) for q in SFs])

    argmax = np.unravel_index(
        np.argmax(q_values_init_state), q_values_init_state.shape)

    value_init_state = np.max(q_values_init_state)
    print(f'Value for init_state at {init_state}', value_init_state)
    print("Optimal action at init_state after composing:", actions[argmax[-1]])

    # The value for going to the RIGHT is 0.7737809040955714 which is exactly .95 ** (N-1) for N = 6 that is the length of the
    # optimal path.

    """
    References
    ----------
    [1] Barreto, André et al. "Successor Features for Transfer in Reinforcement Learning".
    """
