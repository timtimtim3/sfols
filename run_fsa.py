import envs
import gym
import pickle as pkl
import os, shutil
import numpy as np
import wandb
from time import sleep
import argparse 
from envs.wrappers import GridEnvWrapper
from rl.task_specifications import load_fsa
from rl.planning import SFFSAValueIteration as ValueIteration
from datetime import datetime
from rl.utils.utils import seed_everything

def get_successor_features(dirpath):

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)
    return SFs


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


def main() -> None:


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--task", type=str, default="task1")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()
    num_iters = args.num_iters
    task = args.task
    run_name = args.run_name
    seed = args.seed

    seed_everything(seed)

    tmpdir = ".tmp" + str(datetime.now())

    api = wandb.Api()

    # Retrieve run details from cloud and load SFs

    runs =  api.runs(path="davidguillermo/sfcomp")
    olderrun = list(filter(lambda run: str(run.name) == run_name, runs))[0]
    config = olderrun.config

    gym_name = config.pop("env").pop("gym_name")

    os.makedirs(tmpdir, exist_ok=True)

    for file in olderrun.files():

        if file.name.startswith("policies"):
            file.download(root=tmpdir, replace=True)

    sfs = get_successor_features(os.path.join(tmpdir, f"policies/{gym_name}"))
    
    # Load environment & fsa
    eval_env_name = config.pop("eval_env")
    task_name = '-'.join([eval_env_name, task])
    fsa = load_fsa(task_name)

    env = gym.make(eval_env_name)
    env = GridEnvWrapper(env, fsa)


    # 
    newconfig = {
        "num_iters": num_iters,
        "policies_run_name": run_name,
    }

    run = wandb.init(
        config=newconfig,
        entity="davidguillermo", 
        project="sfcomp",
        tags=["sf-vi"]
    )


    # Get environment and FSA
    planning = ValueIteration(env.env, fsa, sfs)
    W = None
    times = [0]
    for i in range(num_iters):

        W, times_ = planning.traverse(W, k=1)

        time = times_[-1]
        times.append(time)
        
        acc_reward = evaluate(env, sfs, W, num_steps=200)


        run.log({'metrics/evaluation/time': np.sum(times),
                'metrics/evaluation/acc_reward': acc_reward,
                'metrics/evaluation/iter': i})

    shutil.rmtree(tmpdir)
    wandb.finish()


if __name__ == "__main__":
    main()

