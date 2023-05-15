import numpy as np
import gym
import wandb as wb
from rl.successor_features.ols import OLS
from rl.utils.utils import eval_test_tasks, hypervolume, policy_evaluation_mo, random_weights
from rl.successor_features.tabular_sf import SF
from rl.successor_features.gpi import GPI
import envs
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle as pkl


if __name__ == "__main__":


    env = gym.make("SimpleRoom-v0")
    eval_env = gym.make("SimpleRoom-v0")

    # These base values are needed to represent the SF at the `terminal` states
    # base_values = {(2, 0, 1, 0): np.asarray([2*[[1, 0]]][0]),
    #                (2, 4, 0, 1): np.asarray([2*[[0, 1]]][0]),
    # }

    base_values = {}

    agent_constructor = lambda: SF(env,
                                alpha=0.3,
                                gamma=0.95,
                                initial_epsilon=0.9,
                                final_epsilon=0.01,
                                epsilon_decay_steps=1000,
                                use_replay=True,
                                per=True,
                                use_gpi=True,
                                envelope=False,
                                batch_size=5,
                                buffer_size=1000000,
                                project_name='SimpleRoom-SFOLS',
                                log=False, 
                                base_values=base_values)
    
    gpi_agent = GPI(env,
                    agent_constructor,
                    log=False,
                    project_name='SimpleRoom-SFOLS',
                    experiment_name="SFOLS_")


    # Number of shapes
    M = 4

    ols = OLS(m=M, epsilon=0.01, reverse_extremum=True)
    test_tasks = random_weights(dim=M, seed=42, n=30) + ols.extrema_weights()
    max_iter = 30

    for iter in range(max_iter):
        w = ols.next_w()
        print('next w', w) 

        gpi_agent.learn(total_timesteps=1000,
                        use_gpi=True,
                        w=w,
                        eval_env=eval_env,
                        eval_freq=100,
                        reset_num_timesteps=False,
                        reset_learning_starts=True,
                        reuse_value_ind=ols.get_set_max_policy_index(w))
                    
        value = policy_evaluation_mo(gpi_agent, eval_env, w, rep=5)
        remove_policies = ols.add_solution(value, w, gpi_agent=gpi_agent, env=eval_env) 

        gpi_agent.delete_policies(remove_policies)

        #print("CCS", ols.ccs)

        returns = [policy_evaluation_mo(gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks]
        returns_ccs = [policy_evaluation_mo(gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in ols.ccs_weights]
        mean_test = np.mean([np.dot(psi, w) for (psi, w) in zip(returns, test_tasks)], axis=0)
        mean_test_smp = np.mean([ols.max_scalarized_value(w_test) for w_test in test_tasks], dtype=np.float64)

        if ols.ended():
            print("ended at iteration", iter)
            for i in range(ols.iteration + 1, max_iter + 1):
                pass
            break
    
    print(len(ols.css))
    for i, pi in enumerate(gpi_agent.policies):

        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")

        with open(f"policies/simpleroom/discovered_policy_{i+1}.pkl", "wb") as fp:

            pkl.dump(d, fp)


    #gpi_agent.close_wandb()