import gym
import hydra
from omegaconf import DictConfig, OmegaConf
from rl.successor_features.ols import OLS
from rl.utils.utils import seed_everything, policy_eval_exact
from rl.successor_features.gpi import GPI
import pickle as pkl
import os
import shutil
import wandb
from envs.wrappers import FlatQEnvWrapper
from rl.task_specifications import load_fsa
from rl.successor_features.tabular_sf import SF
import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group="flatq", tags=["flatq"],

    )
    run.tags = run.tags  # + (cfg.wandb.tag,)
    # Set seeds
    seed_everything(cfg.seed)

    # Create environment for training
    env_params = dict(cfg.env)
    gym_name = env_params.pop("gym_name")
    env = gym.make(gym_name, **env_params)
    test_env = gym.make(gym_name, **env_params)


    # Get task
    train_env = gym.make(cfg.eval_env, **env_params)
    task = cfg.fsa_name

    task_name = '-'.join([train_env.unwrapped.spec.id, task])
    fsa = load_fsa(task_name)  # Load FSA
    train_env = FlatQEnvWrapper(train_env, fsa)
    test_env = FlatQEnvWrapper(test_env, fsa, eval_mode=True)

    # For saving
    directory = env.unwrapped.spec.id
    shutil.rmtree(f"policies/{directory}", ignore_errors=True)
    os.makedirs(f"policies/{directory}", exist_ok=True)

    algo = hydra.utils.call(config=cfg.algorithm, env=train_env)
    algo.learn(total_timesteps=cfg.total_timesteps, total_episodes=cfg.total_episodes, w=np.array([1.0,]), fsa_env=test_env, eval_freq=cfg.eval_freq)



if __name__ == "__main__":
    main()
