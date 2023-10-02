from collections import deque, defaultdict 
import numpy as np

class SFFSAValueIteration:

    def __init__(self, env, fsa, sfs) -> None:
        
        self.env =env 
        self.fsa =fsa 
        self.sfs =sfs 

    def traverse(self, initial_node, weights, k=10000):

        frontier = deque()
        frontier.append(initial_node)
        U = list()

        exit_states = self.env.unwrapped.exit_states

        while len(frontier):
            n = frontier.popleft()
            U.append(n)
            next = [v for v in self.fsa.graph.neighbors(n)]
            for ns in next:
                if ns not in frontier:
                    frontier.append(ns)
        if weights is None:       
            W = np.zeros((len(U), len(exit_states)))
        else:
            W = np.asarray(list(weights.values()))
        
        for _ in range(k):

            W_ = W.copy()
            
            for u in U:
                if self.fsa.is_terminal(u):
                    continue
                for v in self.fsa.get_neighbors(u):
                    # Get the predicates satisfied by the transition
                    predicate = self.fsa.get_predicate((u, v)) 
                    idxs = self.fsa.symbols_to_phi[predicate]

                    uidx, vidx = U.index(u), U.index(v)
                    
                    if self.fsa.is_terminal(v): 
                        W[uidx][idxs] = 1
                    else:
                        for idx in idxs:
                            e = exit_states[idx]
                            W[uidx][idx] = np.asarray([np.dot(q[e], W[vidx]) for q in self.sfs]).max()

            if np.allclose(W, W_):
                break

        W = {u: W[U.index(u)] for u in U}

        return W