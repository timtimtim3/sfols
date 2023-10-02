import numpy as np
import gym
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb as wb
from rl.successor_features.ols import OLS
from rl.utils.utils import policy_evaluation_mo, random_weights, seed_everything, policy_eval_exact
from rl.successor_features.tabular_sf import SF
from rl.successor_features.gpi import GPI
from rl.successor_features.qvalue_iteration import QValueIteration
import envs
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import os
import shutil

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    
    env_params = dict(cfg.env)

    gym_name = env_params.pop("gym_name")

    env = gym.make(gym_name, **env_params)
    eval_env = gym.make(gym_name, **env_params)

    directory = env.unwrapped.spec.id

    shutil.rmtree(f"learned_policies/{directory}", ignore_errors=True)

    os.makedirs(f"learned_policies/{directory}", exist_ok=True)

    # def agent_constructor(): return SF(env, **cfg.algorithm)

    def agent_constructor():
        return hydra.utils.call(config=cfg.algorithm, env=env)

    gpi_agent = GPI(env,
                    agent_constructor,
                    **cfg.gpi.init)

    # Number of shapes
    ols = OLS(m=env.feat_dim, **cfg.ols)
    max_iter = cfg.max_iter

    for iter in range(max_iter):
        w = ols.next_w()
        print('next w', w)

        gpi_agent.learn(w=w,
                        eval_env=eval_env,
                        reuse_value_ind=ols.get_set_max_policy_index(w),
                        **cfg.gpi.learn
                        )

        value = policy_eval_exact(agent=gpi_agent, env=eval_env, w=w)

        # value = policy_evaluation_mo(gpi_agent, eval_env, w, rep=5)  # Do the expectation analytically
        remove_policies = ols.add_solution(
            value, w, gpi_agent=gpi_agent, env=eval_env)

        gpi_agent.delete_policies(remove_policies)

        # print("CCS", ols.ccs)

        # returns = [policy_evaluation_mo(
        #     gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks]
        # returns_ccs = [policy_evaluation_mo(
        #     gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in ols.ccs_weights]
        # mean_test = np.mean([np.dot(psi, w)
        #                     for (psi, w) in zip(returns, test_tasks)], axis=0)
        # mean_test_smp = np.mean([ols.max_scalarized_value(w_test)
        #                         for w_test in test_tasks], dtype=np.float64)

        if ols.ended():
            print("ended at iteration", iter)
            # print(len(ols.ccs), gpi_agent.super().epsilon)
            for i in range(ols.iteration + 1, max_iter + 1):
                pass

            break
    for i, pi in enumerate(gpi_agent.policies):

        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")
        d["w"] = ols.ccs[i]


        with open(f"learned_policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl", "wb") as fp:

            pkl.dump(d, fp)

    # gpi_agent.close_wandb()

if __name__ == "__main__":
    main()