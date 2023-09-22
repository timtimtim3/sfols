from rl.rm import FiniteStateAutomaton
import envs 
import gym
import os
import pickle as pkl
from rl.planning import SFFSAValueIteration as ValueIteration
from rl import rm
import numpy as np

def _get_successor_features(dirpath):

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)
    return SFs


if __name__ == "__main__":

    # Read the successor features
    sfs = _get_successor_features("policies/DeliveryMini-v0")


    # Instantiate the FSA
    symbols_to_phi = {"A": [0], "B":[1], "H":[2]}
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "B")
    fsa.add_transition("u2", "u3", "H")

    env = gym.make("DeliveryMini-v0")

    env.reset()

    s = env.unwrapped.reset((7,3))


    planning = ValueIteration(env, fsa, sfs)

    W = planning.traverse("u0")

    w = np.asarray([0, 1, 0])

    print(w)

    V = np.zeros((8, 8))

    while True:

        state = tuple(s)
        qvalues = np.asarray([np.dot(q[state], w) for q in sfs])

        action = np.unravel_index(qvalues.argmax(), qvalues.shape)
        s, reward, done, phi = env.step(action[1])

        print(state, action[1], qvalues[action])

        if done:
            break

      


    # print(np.round(V, 3))