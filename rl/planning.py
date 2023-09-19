from collections import deque, defaultdict 
import numpy as np

class PlanningFSA:

    def __init__(self, env, fsa, sfs) -> None:
        
        self.env =env 
        self.fsa =fsa 
        self.sfs =sfs 

    def traverse(self, initial_node):

        frontier = deque()
        frontier.append(initial_node)
        Q = list()

        exit_states =self.env.unwrapped.exit_states

        while len(frontier):
            n = frontier.popleft()
            Q.append(n)
            next = [v for v in self.fsa.graph.neighbors(n)]
            for ns in next:
                if ns not in frontier:
                    frontier.append(ns)
                
        W = np.zeros((len(Q), len(exit_states)))

        
        for _ in range(5):
            
            for u in Q:
                if self.fsa.is_terminal(u):
                    continue
                for v in self.fsa.get_neighbors(u):
                    # Get the predicates satisfied by the transition
                    predicate = self.fsa.get_predicate((u, v)) 
                    idxs = self.fsa.symbols_to_phi[predicate]

                    uidx, vidx = Q.index(u), Q.index(v)
                    

                    if self.fsa.is_terminal(v): 
                        W[uidx][idxs] = 1
                    else:
                        for idx in idxs:
                            e = exit_states[idx]
                            W[uidx][idx] = np.asarray([np.dot(q[e], W[vidx]) for q in self.sfs]).max()


        
        W = {u: W[Q.index(u)] for u in Q}

        return W