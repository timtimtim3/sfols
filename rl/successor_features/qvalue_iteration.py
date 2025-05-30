

from typing import Union
import torch as th
import wandb

from sfols.rl.rl_algorithm import RLAlgorithm
from sfols.rl.successor_features.gpi import GPI
import numpy as np


class QValueIteration(RLAlgorithm):


    def __init__(self,  
                env,
                fsa_env = None,
                gamma: float = 0.95,
                delta: float = 1e-3, 
                gpi: GPI = None,
                use_gpi: bool = False,
                log: bool = False,
                log_prefix: str = "",
                constraint: dict = None,
                ):
        
        super().__init__(env, fsa_env=fsa_env, device=None, log_prefix=log_prefix)
        self.phi_dim = len(env.unwrapped.w)
        self.gamma = gamma 
        self.delta = delta
        self.gpi = gpi 
        self.use_gpi = use_gpi
        self.log = log

        self.replay_buffer = None

        self.q_table = dict()

        self.constraint = constraint

    
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
            
            q_values = np.dot(self.q_table[obs], w)
            idxs = np.argwhere(q_values == np.max(q_values)).flatten()

            return np.random.choice(idxs)
        

    def q_values(self, obs: np.array, w: np.array) -> np.array:
        obs = tuple(obs)

        if obs not in self.q_table:
            self.q_table[obs] = np.zeros((self.action_dim, self.phi_dim))
        return np.dot(self.q_table[obs], w)
    

    def train(self, w: np.array, is_option: bool = False):
        s_dim = len(self.env.state_to_coords)
        a_dim = self.env.action_space.n
        phi_dim = self.phi_dim
        Psi_sf = np.zeros(shape=(s_dim, a_dim, phi_dim), dtype=np.float32)

        total_sweeps = 0
        To = None
        if is_option:
            To = np.zeros((s_dim, s_dim)) 
            ts = np.where(w==1)[0].item()
            exit_state = self.env.exit_states[ts]
            tsidx = self.env.coords_to_state[exit_state]

        while True:
            
            total_sweeps += 1
            Psi_new = np.zeros_like(Psi_sf)

            for s in range(s_dim):
                coords = self.env.state_to_coords[s]
                all_ns = []
                for a in range(a_dim):
                    q = 0
                    for ns in range(s_dim):
                        prob = self.env.P[s, a, ns]
                        if not prob:
                            continue
                        all_ns.append(ns)
                        features = self.env.features(self.env.state_to_coords[s], a, self.env.state_to_coords[ns])
                        done = self.env.is_done(self.env.state_to_coords[s], a, self.env.state_to_coords[ns])
                        b = np.argmax(np.dot(Psi_sf[ns], w))
                        # Change this
                        q += prob * (features + self.gamma * (1-done) * Psi_sf[ns, b])

                        if is_option and ns == tsidx:
                                To[s, ns] = self.gamma ** 0            
                    Psi_new[s, a] = q
                self.q_table[coords] = Psi_new[s, :]
               
                if is_option:
                    To[s, tsidx] = self.gamma * np.max(To[all_ns, tsidx])
            if np.allclose(Psi_sf @ w, Psi_new @ w, atol=1e-7):
                break
            else:
                Psi_sf = Psi_new
        # Probably need to refactor this at some point?
        if is_option:
            return Psi_sf, total_sweeps, To 
        else:
            return Psi_sf, total_sweeps, 
    
    def get_config(self) -> dict:
        return {
                'gamma': self.gamma,
                'delta': self.delta,
                }

    def learn(self, w=np.array([1.0,0.0]), **kwargs):
        self.obs, done = self.env.reset(), False
        self.env.unwrapped.w = w
        self.w = w

        _, total_sweeps = self.train(w)
        # TODO: Add information about the policy obtained?

        init_state = self.env.unwrapped.initial[0]
        if self.log:
            wandb.log({f"{self.log_prefix}vi iters": total_sweeps})
