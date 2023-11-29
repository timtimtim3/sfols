import envs
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


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project, tags=["sfols"]
    )

    # Set seeds
    seed_everything(cfg.seed)

    # Create env
    env_params = dict(cfg.env)
    gym_name = env_params.pop("gym_name")
    env = gym.make(gym_name, **env_params)
    eval_env = gym.make(gym_name, **env_params)

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
        
    for i, pi in enumerate(gpi_agent.policies):
        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")
        with open(f"policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl", "wb") as fp:
            pkl.dump(d, fp)
        wandb.save(f"policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl")


if __name__ == "__main__":
    main()
