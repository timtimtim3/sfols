import numpy as np
import gym
import wandb as wb
from rl.successor_features.ols import OLS
from rl.utils.utils import eval_test_tasks, hypervolume, policy_evaluation_mo, random_weights, seed_everything
from rl.successor_features.tabular_sf import SF
from rl.successor_features.qvalue_iteration import QValueIteration
from rl.successor_features.gpi import GPI
import envs
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle as pkl
import os
from envs.rm import * 
from rl.rm_constructor import *

# import hydra 
# from omegaconf import DictConfig, OmegaConf 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-env')
    args = parser.parse_args()
    env_name = args.env

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    envid = env.unwrapped.spec.id
    
    # Step 1: Learn the 'Option's policies' 
    num_targets = len(env.unwrapped.exit_states)
    weights = np.eye(num_targets)

    
    # Instatiate the GPI-QVI algorithm
    agent_constructor = lambda : QValueIteration(env, use_gpi=True, gamma=0.99)

    Sh, Sw = env.unwrapped.height, env.unwrapped.width
    Sdim = env.unwrapped.s_dim

    print(f"Env-id: {envid}")

    OPTIONS = {}

    for i, w in enumerate(weights):

        print(f"Training for {w}")

        policy = QValueIteration(env, use_gpi=True, gamma=0.99)
        _, total_iters, To = policy.train(w, is_option=True)

        # Save the policy
        d = vars(policy)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")
        d["w"] = w

        with open(f"policies/lof/{envid}/discovered_policy_{i+1}.pkl", "wb") as fp:

            pkl.dump(d, fp)

        # Step 1.5: From SFs policies to options as in LOF paper
        states = np.ndindex((Sh, Sw))

        qvalues = np.asarray([np.dot(policy.q_table[tuple(s)], w) for s in states])
    
        Ro = qvalues.max(axis=1)

        option = {'qfn': qvalues, 'To': To, 'Ro' : Ro}

        OPTIONS[i] = option
    
    # Step 2: Use VI for learning the Values on FSA
    # NOTE: We are ignoring event are trivial in the fully observable setting 
    # (ack by the authors section 3 - propositions part).

    fsa = fsa_delivery_mini1()
    F = fsa.states
    exit_states = env.unwrapped.exit_states

    Q = np.zeros((len(F), Sdim, len(OPTIONS)))
    V = np.zeros((len(F), Sdim))

    # LOF-VI

    iters = 0

    while True:

        iters+=1

        Q_old = Q.copy()
        V_old = V.copy()

        for fidx, sidx in np.ndindex((len(F), Sdim)):
            
            f = F[fidx]

            if f == 'u3':
                break

            for o in range(len(OPTIONS)):

                Qo = OPTIONS[o]['qfn']
                To = OPTIONS[o]['To']
                Ro = OPTIONS[o]['Ro']

                # Eq. 3 LOF paper
                next_fsa_states = fsa.get_neighbors(F[fidx])
                
                next_exit_state = exit_states[o]

                aux = [To[sidx, np.ravel_multi_index(next_exit_state, (Sh, Sw))]* V_old[F.index(nf), np.ravel_multi_index(next_exit_state, (Sh, Sw))] for nf in next_fsa_states]
                            
                Q[fidx, sidx, o] = Ro[sidx] + np.sum(aux)
        
        V = Q.max(axis=2)

        if np.allclose(V, V_old):
            print("Done", iters)
            break


    mu = Q.argmax(axis=2)

    for f, s in np.ndindex(mu.shape):
        if f == 3:
            break
        state =  np.unravel_index(s, (Sh, Sw))
        print(F[f], state, env.unwrapped.MAP[state],  mu[f, s], Q[f, s, :])
        


