from rm import FiniteStateAutomaton
import networkx as nx
import numpy as np
from collections import deque 
import envs 
from collections import defaultdict
import gym
import os
import pickle as pkl

def _get_successor_features(dirpath):

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)

    return SFs

def traverse_fsa(fsa, env, sfs, initial_node):
    
    # Interpret the last node
    frontier = deque()
    frontier.append(initial_node)
    Q = deque()

    exit_states = env.unwrapped.exit_states

    while len(frontier):
        n = frontier.popleft()
        Q.append(n)
        next = [v for v in fsa.graph.neighbors(n)]
        for ns in next:
            if ns not in frontier:
                frontier.append(ns)
    
    
    W = defaultdict(lambda: np.zeros(len(exit_states)))

    Q.popleft()
    
    while len(Q):
        state = Q.pop()

        for (u, v) in fsa.in_transitions(state):
            predicate = fsa.get_predicate((u, v))
            idxs = fsa.symbols_to_phi[predicate]
    
            if not len(W):    
                W[u][idxs] = 1
            else:
                for idx in idxs:
                    e = exit_states[idx]
                    print(state, e, np.asarray([np.dot(q[e], W[v]) for q in sfs]).max(), env.unwrapped.PHI_OBJ_TYPES[idx])
                    W[u][idx] += np.asarray([np.dot(q[e], W[v]) for q in sfs]).max()



    print(W)



if __name__ == "__main__":

    fsa = FiniteStateAutomaton()

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

    fsa.symbols_to_phi ={"COFFEE": [0, 1], "OFFICE":[2, 3], "MAIL": [4]}

    edges = fsa.in_transitions("u4")


    env = gym.make("OfficeComplex-v0")
    
    sfs = _get_successor_features("policies/OfficeComplex-v0")

    traverse_fsa(fsa, env, sfs, "u0")
