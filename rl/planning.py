from collections import deque, defaultdict 
import numpy as np
import time as time

class SFFSAValueIteration:

    def __init__(self, env, fsa, sfs) -> None:
        
        self.env =env 
        self.fsa =fsa 
        self.sfs =sfs 

    def traverse(self, weights, k=10000):

        exit_states = self.env.unwrapped.exit_states

        U = self.fsa.states
        
        if weights is None:       
            W = np.zeros((len(U), len(exit_states)))
        else:
            W = np.asarray(list(weights.values()))

        
        times = [0]

        
        for _ in range(k):

            start_iter = time.time()

            W_ = W.copy()
            
            for u in U:
                
                if self.fsa.is_terminal(u):
                    continue
                for v in self.fsa.get_neighbors(u):
                    
                    # Get the predicates satisfied by the transition
                    propositions = self.fsa.get_predicate((u, v)) 
                    idxs = [self.fsa.symbols_to_phi[prop] for prop in propositions]

                    uidx, vidx = U.index(u), U.index(v)
                    
                    if self.fsa.is_terminal(v): 
                        W[uidx][idxs] = 1
                    else:
                        for idx in idxs:
                            e = exit_states[idx]
                            W[uidx][idx] = np.asarray([np.dot(q[e], W[vidx]) for q in self.sfs]).max()
            
            elapsed_time = time.time() - start_iter
            times.append(elapsed_time)

            if np.allclose(W, W_):
                break

        W = {u: W[U.index(u)] for u in U}


        times = np.cumsum(times)

        return W, times