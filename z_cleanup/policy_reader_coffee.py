import pickle as pkl
import numpy as np
import os
import envs
import gym
# import argparse
import matplotlib.pyplot as plt
from plotting.plotting import create_grid_plot, plot_policy, get_plot_arrow_params



actions = ["LEFT", "UP", "RIGHT", "DOWN"]

# Enumerate the relevant locations (check the environment)
coffee1 = (1, 2)
coffee2 = (2, 5)
office2 = (0, 5)
initial = (4, 2)

env = gym.make("CoffeeOffice-v1") 

print(env.unwrapped.object_ids)
print(env.unwrapped.all_objects)
print(env.unwrapped.initial)


def _get_plot_arrow_params(q_table, w):
    x_pos = []
    y_pos = []
    x_dir = []
    y_dir = []
    color = []

    for coords,q_vals in q_table.items():
        max_val = np.max(q_vals)
        max_index = np.argmax(q_vals)

        print(coords, q_vals, max_index)

        x_d = y_d = 0
        if max_index == 3:
            y_d = 1
        elif max_index == 1:
            y_d = -1
        elif max_index == 2:
            x_d = 1
        elif max_index == 0:
            x_d = -1

        x_pos.append(coords[1] + 0.5)
        y_pos.append(coords[0] + 0.5)
        x_dir.append(x_d)
        y_dir.append(y_d)
        color.append(max_val)
    # down, up , right, left
    return np.array(x_pos), np.array(y_pos), np.array(x_dir), np.array(y_dir), np.array(color)


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

    SFs = _get_successor_features(os.path.abspath("policies/CoffeeOffice-v0"))

    # 2) We set the weights to 1 for each of the offices as any of the sastisfies the task,
    # we compute the q-function for each policy and for each 'coffee' location and
    # then we can apply GPI (see Theorem 1 in [1])

    weights_office = np.asarray([0, 0, 1, 1])

    q_values_coffee1 = np.asarray(
        [np.dot(q[coffee1], weights_office) for q in SFs])
    

    q_values_coffee2 = np.asarray(
        [np.dot(q[coffee2], weights_office) for q in SFs])
    
    q_values_office2 = np.asarray(
        [np.dot(q[office2], weights_office) for q in SFs])
    

    print(np.max(q_values_coffee1, axis=0))

    value_coffee1 = np.max(q_values_coffee1)
    value_coffee2 = np.max(q_values_coffee2)
    print(f'Value for coffee 1 at {coffee1}', value_coffee1)
    print(f'Value for coffee 2 at {coffee2}', value_coffee2)

    # exit()

    # Such values there after are the weights to combine the coffee subtask.
    weights_coffee = np.asarray([value_coffee1, value_coffee2, 0, 0])

    SFs = _get_successor_features(os.path.abspath("policies/CoffeeOffice-v0"))

    q_values_init_state = np.asarray(
        [np.dot(q[initial], weights_coffee) for q in SFs])

    argmax = np.unravel_index(
        np.argmax(q_values_init_state), q_values_init_state.shape)

    value_init_state = np.max(q_values_init_state)
    print(f'Value for init_state at {initial}', value_init_state)
    print("Optimal action at init_state after composing:", actions[argmax[-1]])

    # The value for going to the RIGHT is 0.7737809040955714 which is exactly .95 ** (N-1) for N = 6 that is the length of the
    # optimal path.


def coffee_and_mail_then_office_deterministic():

    # 1) Get the SF representstion for the policies in the CCS for the office task.

    SFs = _get_successor_features(os.path.abspath("policies/CoffeeOffice-v1"))

    # 2) We set the weights to 1 for each of the offices as any of the sastisfies the task,
    # we compute the q-function for each policy and for each 'coffee' location and
    # then we can apply GPI (see Theorem 1 in [1])

    coffee = (1, 2)
    mail = (2, 5)
    office1 = (5, 0)
    office2 = (0, 5)

    initial = (4, 2)

    weights = np.asarray([1, 0, 0, 1])

    def get_value_state(state:tuple):

        q  = np.round(np.asarray([np.dot(q[state], weights) for q in SFs]).max(axis=0), 4)
        
        value_state = np.round(np.max(q), 4)

        print(f'Q({state}) = ', q)
        print(f'V({state}) = ', value_state )
        opt_actions = np.where(np.isclose(q, value_state, atol=1e-3))[0].tolist()
        print(f'PI({state})', list(map(lambda x: actions[x], opt_actions)), "\n")

        return q


    q_table = {}

    for (i, j) in env.unwrapped.coords_to_state:
        state = (i, j)
        q_table[state] = get_value_state(state)

    plt.figure(1)

    ax = plt.subplot((1+1)*100 + 11)
    ax.set_title(f"w={weights}")

    create_grid_plot(ax=ax, grid=env.MAP != 'X')

    w =  np.ones(4)

    quiv = plot_policy(
                ax=ax, arrow_data= _get_plot_arrow_params(q_table, w), grid=env.MAP,
                values=False, max_index=False
            )
    # finish test
    plt.show()


    exit()

    q_values_coffee = np.asarray(
        [np.dot(q[coffee], weights_office) for q in SFs])
    
    q_values_mail = np.asarray(
        [np.dot(q[mail], weights_office) for q in SFs])
    
    gamma = 0.95

    value_coffee_after_mail = np.max(q_values_coffee)
    value_mail_after_coffee = np.max(q_values_mail)
    print(f'Value at U4 for COFFE at {coffee}', value_coffee_after_mail)
    print(f'Value at U4 for MAIL at {mail}', value_mail_after_coffee)

    # exit()

    # Such values there after are the weights to combine the coffee subtask.
    weights_coffee = np.asarray([0, value_mail_after_coffee, 0, 0])
    weights_mail = np.asarray([value_coffee_after_mail, 0, 0, 0])

    q_values_coffee = np.asarray(
        [np.dot(q[coffee], weights_coffee) for q in SFs])
    
    q_values_mail = np.asarray(
        [np.dot(q[mail], weights_mail) for q in SFs])

    value_coffee =  gamma * np.max(q_values_coffee)
    value_mail = gamma * np.max(q_values_mail)

    weights_mail_or_coffee = np.asarray([value_coffee, value_mail, 0, 0])

    q_values_init_state = gamma * np.asarray(
        [np.dot(q[initial], weights_mail_or_coffee) for q in SFs])

    argmax = np.unravel_index(
        np.argmax(q_values_init_state), q_values_init_state.shape)

    value_init_state = np.max(q_values_init_state)
    print(f'Value for init_state at {initial}', value_init_state)
    print("Optimal action at init_state after composing:", actions[argmax[-1]])

    # The value for going to the RIGHT is 0.7737809040955714 which is exactly .95 ** (N-1) for N = 6 that is the length of the
    # optimal path.


def office2_no_coffee():

    # 1) Get the SF representstion for the policies in the CCS for the office task.

    SFs = _get_successor_features(os.path.abspath("policies/CoffeeOffice-v0"))

    # 2) We set the weights to 1 for each of the offices as any of the sastisfies the task,
    # we compute the q-function for each policy and for each 'coffee' location and
    # then we can apply GPI (see Theorem 1 in [1])

    coffee1 = (1, 2)
    coffee2 = (2, 5)
    office2 = (0, 5)
    initial = (4, 2)


    weights = np.asarray([1, 1, 0, 0])
    
    gamma = 0.95

    q_values_init_state = gamma * np.asarray(
        [np.dot(q[initial], weights) for q in SFs])
    

    #print(q_values_init_state)

    argmax = np.unravel_index(
        np.argmax(q_values_init_state), q_values_init_state.shape)

    # Such values there after are the weights to combine the coffee subtask.
    value_init_state = np.max(q_values_init_state)
    print(f'Value for init_state at {initial}', value_init_state)
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

    # coffee_and_mail_then_office_deterministic()
    coffee_and_mail_then_office_deterministic()

"""
References
----------
[1] Barreto, André et al. "Successor Features for Transfer in Reinforcement Learning".
"""
