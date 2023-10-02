from rl.fsa import FiniteStateAutomaton
import envs 
import gym
import os
import pickle as pkl
from rl.planning import SFFSAValueIteration as ValueIteration
from rl import fsa

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
    sfs = _get_successor_features("policies/OfficeComplex-v0")


    # Instantiate the FSA
    symbols_to_phi = {"COFFEE": [0, 1], "OFFICE":[2, 3], "MAIL": [4, 5], "DECORATION":[6, 7]}
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "COFFEE")
    fsa.add_transition("u0", "u2", "MAIL")
    fsa.add_transition("u1", "u3", "MAIL")
    fsa.add_transition("u2", "u3", "COFFEE")
    fsa.add_transition("u3", "u4", "OFFICE")

    env = gym.make("OfficeComplex-v0")
    

    planning = ValueIteration(env, fsa, sfs)

    W = planning.traverse("u0")
    print(W)


