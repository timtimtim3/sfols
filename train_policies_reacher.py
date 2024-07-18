import numpy as np
import gym
#from gym.wrappers import Monitor
import wandb
from rl.utils.utils import eval_test_tasks, hypervolume, moving_average, random_weights, policy_evaluation_mo
from rl.successor_features.sf_dqn import SFDQN
from rl.successor_features.gpi import GPI
from rl.successor_features.ols import OLS
import envs
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def main(algo):

    run = wandb.init(
        config={},
        entity="guillermoim", project="SFOLS-reacher",
        group="one-hot-0.05", 
        tags=["sfols"],
        mode="disabled"

    )

    env = gym.make("ReacherMultiTask-v0")
    #eval_env = Monitor(gym.make("ReacherMultiTask-v0"), 'videos/', video_callable=lambda i: i % 10 == 0)
    eval_env = gym.make("ReacherMultiTask-v0")

    agent_constructor = lambda log_prefix: SFDQN(env,
                                    gamma=0.9,
                                    net_arch=[256, 256],
                                    learning_rate=1e-2,
                                    batch_size=256,
                                    initial_epsilon=0.5,
                                    final_epsilon=0.05,
                                    epsilon_decay_steps=100,
                                    per=True,
                                    min_priority=0.01,
                                    buffer_size=int(4e6),
                                    gradient_updates=1,
                                    tau=1.0,
                                    target_net_update_freq=500,
                                    log=False)
    agent = GPI(env,
                agent_constructor,
                experiment_name=algo)

    ols = OLS(m=4, epsilon=0.01, reverse_extremum=True)
    test_tasks = random_weights(dim=4, seed=42, n=60) + ols.extrema_weights()
    max_iter = 15
    
    for iter in range(max_iter):
        w = ols.next_w()
        
        print('next w', w)

        agent.learning_starts = agent.num_timesteps # reset epsilon exploration
        agent.learn(total_timesteps=500000,
                    w=w,
                    eval_env=eval_env,
                    eval_freq=1000,
                    reuse_value_ind=None)

        value = policy_evaluation_mo(agent, eval_env, w, rep=5)
        remove_policies = ols.add_solution(value, w, gpi_agent=agent, env=eval_env)
        agent.delete_policies(remove_policies)     

        returns = [policy_evaluation_mo(agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks]
        returns_ccs = [policy_evaluation_mo(agent, eval_env, w, rep=5, return_scalarized_value=False) for w in ols.ccs_weights]
        mean_test = np.mean([np.dot(psi, w) for (psi, w) in zip(returns, test_tasks)], axis=0)
        wandb.log({'eval/mean_value_test_tasks': mean_test, 'iteration': ols.iteration})
        mean_test_smp = np.mean([ols.max_scalarized_value(w_test) for w_test in test_tasks])
        wandb.log({'eval/mean_value_test_tasks_SMP': mean_test_smp, 'iteration': ols.iteration}) 
        wandb.log({'eval/hypervolume': hypervolume(np.zeros(4), ols.ccs), 'iteration': ols.iteration})
        wandb.log({'eval/hypervolume_GPI': hypervolume(np.zeros(4), returns+returns_ccs), 'iteration': ols.iteration})

        if ols.ended():
            print("ended at iteration", iter)
            for i in range(ols.iteration + 1, max_iter + 1):
                wandb.log({'eval/mean_value_test_tasks': mean_test, 'iteration': i})
                wandb.log({'eval/mean_value_test_tasks_SMP': mean_test_smp, 'iteration': i})
                wandb.log({'eval/hypervolume': hypervolume(np.zeros(4), ols.ccs), 'iteration': i}) 
                wandb.log({'eval/hypervolume_GPI': hypervolume(np.zeros(4), returns+returns_ccs), 'iteration': i})
            break
    
    #agent.save()

    agent.close_wandb()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reacher experiment.')
    parser.add_argument('-algo', type=str, choices=['SFOLS', 'WCPI', 'Random'], default='SFOLS', help='Algorithm.')
    args = parser.parse_args()

    main(args.algo)
