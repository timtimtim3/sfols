import envs
import gym
import hydra
from omegaconf import DictConfig, OmegaConf
from rl.successor_features.ols import OLS
from rl.utils.utils import seed_everything, policy_eval_exact, policy_evaluation_mo
from rl.successor_features.gpi import GPI
import pickle as pkl
import os
import shutil
import wandb
from envs.wrappers import GridEnvWrapper
from rl.task_specifications import load_fsa
from rl.planning import SFFSAValueIteration as ValueIteration
import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group="sfols", tags=["sfols"],

    )
    run.tags = run.tags #+ (cfg.wandb.tag,)
    # Set seeds
    seed_everything(cfg.seed)

    # Create environment for training
    env_params = dict(cfg.env)
    gym_name = env_params.pop("gym_name")
    env = gym.make(gym_name, **env_params)
    test_env = gym.make(gym_name, **env_params)

    # Get task
    eval_env = gym.make(cfg.eval_env)
    task = cfg.fsa_name

    task_name = '-'.join([eval_env.unwrapped.spec.id, task])
    fsa = load_fsa(task_name) # Load FSA
    eval_env = GridEnvWrapper(eval_env, fsa)

    # For saving
    directory = env.unwrapped.spec.id
    shutil.rmtree(f"policies/{directory}", ignore_errors=True)
    os.makedirs(f"policies/{directory}", exist_ok=True)

    def agent_constructor(log_prefix: str):
        return hydra.utils.call(config=cfg.algorithm, env=env, log_prefix=log_prefix)

    gpi_agent = GPI(env,
                    agent_constructor,
                    **cfg.gpi.init)

    # Number of shapes
    ols = OLS(m=env.feat_dim, **cfg.ols)


    for ols_iter in range(cfg.max_iter):
        
        if ols.ended():
            print("ended at iteration", ols_iter)
            break
       
        w = ols.next_w()
        print(f"Training {w}")

        gpi_agent.learn(w=w,
                        reuse_value_ind=ols.get_set_max_policy_index(w),
                        fsa_env = eval_env,
                        eval_env = test_env,
                        **cfg.gpi.learn
                        )

        value = policy_eval_exact(agent=gpi_agent, env=test_env, w=w) # Do the expectation analytically
        # print(f"policy{ols_iter} - exact value", value)
        # value = policy_evaluation_mo(gpi_agent, test_env, w, rep=5) 
        # print(f"policy{ols_iter} -estimated value", value)

        remove_policies = ols.add_solution(
            value, w, gpi_agent=gpi_agent, env=test_env)

        gpi_agent.delete_policies(remove_policies)

        # eval_reward = eval_on_fsa(eval_env, task, gpi_agent.policies)
        
        # wandb.log({"fsa_reward": eval_reward,})
        
    for i, pi in enumerate(gpi_agent.policies):
        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")
        with open(f"policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl", "wb") as fp:
            pkl.dump(d, fp)
        wandb.save(f"policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl")

        run.summary["policies_obtained"] = len(gpi_agent.policies)

if __name__ == "__main__":
    main()
