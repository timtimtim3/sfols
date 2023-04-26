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


def run(algo):

    env = gym.make("SimpleRoom-v0")
    eval_env = gym.make("SimpleRoom-v0")

    agent_constructor = lambda: SF(env,
                                alpha=0.3,
                                gamma=0.95,
                                initial_epsilon=1,
                                final_epsilon=0.05,
                                epsilon_decay_steps=900000,
                                use_replay=True,
                                per=True,
                                use_gpi=True,
                                envelope=False,
                                batch_size=5,
                                buffer_size=1000000,
                                project_name='SimpleRoom-SFOLS',
                                log=False)
    gpi_agent = GPI(env,
                    agent_constructor,
                    log=True,
                    project_name='FourRoom-SFOLS',
                    experiment_name=algo)

    ols = OLS(m=4, epsilon=0.01, reverse_extremum=True)
    test_tasks = random_weights(dim=4, seed=42, n=30) + ols.extrema_weights()
    max_iter = 30

    sip_weights = []
    for i in range(4):
        w = -1 * np.ones(4)
        w[i] = 1
        sip_weights.append(w)

    for iter in range(max_iter):
        if algo == 'SFOLS':
            w = ols.next_w()
        elif algo == 'WCPI':
            w = ols.worst_case_weight()
        elif algo == 'SIP':
            w = sip_weights[iter]
        elif algo == 'Random':
            w = random_weights(dim=4)
        print('next w', w)

        gpi_agent.learn(total_timesteps=100000,
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

        print("CCS", ols.ccs)

        returns = [policy_evaluation_mo(gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks]
        returns_ccs = [policy_evaluation_mo(gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in ols.ccs_weights]
        mean_test = np.mean([np.dot(psi, w) for (psi, w) in zip(returns, test_tasks)], axis=0)
        wb.log({'eval/mean_value_test_tasks': mean_test, 'iteration': ols.iteration})
        mean_test_smp = np.mean([ols.max_scalarized_value(w_test) for w_test in test_tasks], dtype=np.float64)
        wb.log({'eval/mean_value_test_tasks_SMP': mean_test_smp, 'iteration': ols.iteration}) 
        wb.log({'eval/hypervolume': hypervolume(np.zeros(4), ols.ccs), 'iteration': ols.iteration})
        wb.log({'eval/hypervolume_GPI': hypervolume(np.zeros(4), returns+returns_ccs), 'iteration': ols.iteration})

        if ols.ended() or (algo == 'SIP' and iter == 2):
            print("ended at iteration", iter)
            for i in range(ols.iteration + 1, max_iter + 1):
                wb.log({'eval/mean_value_test_tasks': mean_test, 'iteration': i})
                wb.log({'eval/mean_value_test_tasks_SMP': mean_test_smp, 'iteration': i})
                wb.log({'eval/hypervolume': hypervolume(np.zeros(4), ols.ccs), 'iteration': i}) 
                wb.log({'eval/hypervolume_GPI': hypervolume(np.zeros(4), returns+returns_ccs), 'iteration': i})
            break


    for i, pi in enumerate(gpi_agent.policies):

        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")

        with open(f"discovered_policy_{i+1}.pkl", "wb") as fp:

            pkl.dump(d, fp)

       
    with open("discovered_policies.pkl", "wb") as fp:
        pkl.dump(gpi_agent.policies, fp)

    gpi_agent.close_wandb()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Four Room experiment.')
    parser.add_argument('-algo', type=str, choices=['SFOLS', 'WCPI', 'SIP', 'Random'], default='SFOLS', help='Algorithm.')
    args = parser.parse_args()

    run(args.algo)
