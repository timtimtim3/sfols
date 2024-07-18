import gym 
import envs 
import os, wandb, pickle as pkl
from datetime import datetime

from rl.successor_features.gpi import GPI
from rl.successor_features.tabular_sf import SF

import numpy as np


def get_successor_features(dirpath):

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)
    return SFs    



if __name__ == "__main__":

    run_name = "electric-butterfly-1104"
    
    
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

    
    w = np.asarray([1,1,0,0,1])
    
    env = gym.make("DeliveryPenaltyEval-v0")
    state = tuple(env.reset())

    done = False

   
    qvalues = np.asarray([np.dot(q[1, 3], w) for q in sfs])
    action_idx = np.unravel_index(qvalues.argmax(), qvalues.shape)

    print(qvalues, action_idx)    




