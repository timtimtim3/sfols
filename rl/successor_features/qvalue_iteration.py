
from typing import Union
import torch as th
from rl.rl_algorithm import RLAlgorithm
from rl.successor_features.gpi import GPI
import numpy as np



class QValueIteration(RLAlgorithm):


    def __init__(self,  
                env,
                gamma: float = 0.95,
                delta: float = 1e-3, 
                gpi: GPI = None,
                use_gpi: bool = False):
        
        super().__init__(env, device=None)


        self.phi_dim = len(env.unwrapped.w)
        self.gamma = gamma 
        self.delta = delta
        self.gpi = gpi 
        self.use_gpi = use_gpi

        self.q_table = dict()

    
    def act(self, obs: np.array, w: np.array):
        np_obs = obs
        obs = tuple(obs)
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.phi_dim))
        self.policy_index = None
        
        if self.gpi is not None and self.use_gpi:
            action, self.policy_index = self.gpi.eval(np_obs, w, return_policy_index=True)
            return action
        else:
            return np.argmax(np.dot(self.q_table[obs], w))
            
    def eval(self, obs: np.array, w: np.array) -> int:
        if self.gpi is not None and self.use_gpi:
            return self.gpi.eval(obs, w)
        else:
            obs = tuple(obs)
            if obs not in self.q_table:
                return int(self.env.action_space.sample())
            return np.argmax(np.dot(self.q_table[obs], w))
        

    def q_values(self, obs: np.array, w: np.array) -> np.array:
        obs = tuple(obs)

        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.phi_dim))
        return np.dot(self.q_table[obs], w)
    

    def train(self, w: np.array):
        s_dim = len(self.env.state_to_coords)
        a_dim = self.env.action_space.n
        phi_dim = self.phi_dim
        Psi_sf = np.zeros(shape=(s_dim, a_dim, phi_dim), dtype=np.float32)
        for _ in range(20):
            Psi_new = np.zeros_like(Psi_sf)
            for s_old in range(s_dim):
                for a in range(a_dim):
                    q = 0
                    for s_new in range(s_dim):
                        prob = self.env.P[s_old, a, s_new]
                        if not prob:
                            continue
                        features = self.env.features(self.env.state_to_coords[s_old], a, self.env.state_to_coords[s_new])
                        # done = self.is_done(self.state_to_coords[s_old], a, self.state_to_coords[s_new])
                        b = np.argmax(Psi_sf[s_new] @ w)
                        # Change this 
                        q += prob * (features + self.gamma * Psi_sf[s_new, b])
                    Psi_new[s_old, a] = q

            Psi_sf = Psi_new
        # Probably need to refactor this at some point?
        return Psi_sf
    
    def get_config(self) -> dict:
        return {
                'gamma': self.gamma,
                'delta': self.delta,
                }