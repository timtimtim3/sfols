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
import pandas as pd

from rl.successor_features.gpi import GPI
from rl.successor_features.tabular_sf import SF

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


        print(f"step: {_} - state: {(f, state)} - qvalues: {qvalues} - action: {action} - features: {phi} - {done}")
        print(obs)


        acc_reward+=reward

        if done:
            break

    return acc_reward


def main() -> None:

    MAX_ITERS = 50

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="rosy-brook-1087")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_name", type=str, default="results-fsas")



    args = parser.parse_args()
    run_name = args.run_name
    seed = args.seed
    csv_name = args.csv_name

    seed_everything(seed)

    tmpdir = ".tmp" + str(datetime.now())

    api = wandb.Api()

    # Retrieve run details from cloud and load SFs

    runs =  api.runs(path="davidguillermo/sfcomp")
    olderrun = list(filter(lambda run: str(run.name) == run_name, runs))[0]
    config = olderrun.config

    gym_name = config.pop("env").pop("gym_name")
    env = gym.make(gym_name)
    
    gpi = GPI(env, lambda: SF, log=False)
    
    os.makedirs(tmpdir, exist_ok=True)

    for file in olderrun.files():

        if file.name.startswith("policies"):
            file.download(root=tmpdir, replace=True)

    sfs = get_successor_features(os.path.join(tmpdir, f"policies/{gym_name}"))

    for q in sfs:
        sf = SF(env)
        sf.q_table = q
        gpi.policies.append(sf)
    

    # Load environment & fsa
    eval_env_name = config.pop("eval_env")

    all_rewards = []

    for task in ("task1",):

        task_name = '-'.join([eval_env_name, task])

        fsa = load_fsa(task_name)

        eval_env = gym.make(eval_env_name)
        fsa_env = GridEnvWrapper(eval_env, fsa)

        # Get environment and FSA
        planning, W = ValueIteration(fsa_env, gpi), None
        
        res_acc_reward = []
        
        for i in range(MAX_ITERS):

            W, _ = planning.traverse(W, k=1)

            acc_reward = evaluate(fsa_env, sfs, W, num_steps=200)
            res_acc_reward.append(acc_reward)

        print(res_acc_reward)

        all_rewards.append(res_acc_reward)
        # shutil.rmtree(tmpdir)

    rewards = np.vstack(all_rewards)

    df = pd.DataFrame({'iter': list(range(MAX_ITERS)),
                        'mean': rewards.mean(axis=0),
                        'std': rewards.std(axis=0)})

    # df.to_csv(f"{csv_name}.csv")
    # print(df)

    wandb.finish()


if __name__ == "__main__":
    main()