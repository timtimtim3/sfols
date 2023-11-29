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
from envs.wrappers import GridEnvWrapper
from rl.task_specifications import load_fsa
from rl.planning import SFFSAValueIteration as ValueIteration
import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    
    # Init Wandb
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.wandb.entity, project=cfg.wandb.project, tags=["sfols"]
    )

    # Set seeds
    seed_everything(cfg.seed)

    # Create environment for training
    env_params = dict(cfg.env)
    gym_name = env_params.pop("gym_name")
    env = gym.make(gym_name, **env_params)
    test_env = gym.make(gym_name, **env_params)

    # Get task
    eval_env = gym.make(cfg.eval_env)
    task = cfg.task

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
                        eval_env=test_env,
                        reuse_value_ind=ols.get_set_max_policy_index(w),
                        **cfg.gpi.learn
                        )

        value = policy_eval_exact(agent=gpi_agent, env=test_env, w=w)

        # value = policy_evaluation_mo(gpi_agent, eval_env, w, rep=5)  # Do the expectation analytically
        remove_policies = ols.add_solution(
            value, w, gpi_agent=gpi_agent, env=test_env)

        gpi_agent.delete_policies(remove_policies)

        eval_reward = eval_on_fsa(eval_env, task, gpi_agent.policies)
        
        wandb.log({"fsa_reward": eval_reward,})
        
    for i, pi in enumerate(gpi_agent.policies):
        d = vars(pi)
        d.pop("replay_buffer")
        d.pop("env")
        d.pop("gpi")
        with open(f"policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl", "wb") as fp:
            pkl.dump(d, fp)
        wandb.save(f"policies/{env.unwrapped.spec.id}/discovered_policy_{i + 1}.pkl")



def evaluate(env, sfs, W, num_steps = 100):
    
    env.reset()
    acc_reward = 0

    for _ in range(num_steps):

        (f, state) = env.get_state()
        w = W[f]
        qvalues = np.asarray([np.dot(q[state], w) for q in sfs])

        action = np.unravel_index(qvalues.argmax(), qvalues.shape)[1]
        obs, reward, done, phi = env.step(action)
        acc_reward+=reward

        if done:
            break

    return acc_reward

def eval_on_fsa(env, task, policies):

    task_name = task_name = '-'.join([env.unwrapped.spec.id, task])

    sfs = [policy.q_table for policy in policies]

    fsa = load_fsa(task_name) # Load FSA

    env = GridEnvWrapper(env, fsa)

    planning = ValueIteration(env.env, fsa, sfs)
    W, times_ = planning.traverse(None, k=15)
    acc_reward = evaluate(env, sfs, W, num_steps=200)

    return acc_reward

if __name__ == "__main__":
    main()
