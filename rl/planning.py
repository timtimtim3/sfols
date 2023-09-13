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
        Q = deque()

        exit_states =self.env.unwrapped.exit_states

        while len(frontier):
            n = frontier.popleft()
            Q.append(n)
            next = [v for v in self.fsa.graph.neighbors(n)]
            for ns in next:
                if ns not in frontier:
                    frontier.append(ns)
                
        W = defaultdict(lambda: np.zeros(len(exit_states)))

        Q.popleft()
        
        while len(Q):
            state = Q.pop()

            for (u, v) in self.fsa.in_transitions(state):
                predicate = self.fsa.get_predicate((u, v))
                idxs = self.fsa.symbols_to_phi[predicate]
        
                if not len(W):    
                    W[u][idxs] = 1
                else:
                    for idx in idxs:
                        e = exit_states[idx]
                        W[u][idx] += np.asarray([np.dot(q[e], W[v]) for q in self.sfs]).max()

        return W