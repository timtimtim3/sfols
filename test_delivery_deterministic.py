from rl.fsa import FiniteStateAutomaton
import envs 
import gym
import os
import pickle as pkl
from rl.planning import SFFSAValueIteration as ValueIteration
from rl.task_specifications import *
import numpy as np
from envs.wrappers import GridEnvWrapper

def _get_successor_features(dirpath):

    SFs = []

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)
        q = policy["q_table"]
        SFs.append(q)
    return SFs


if __name__ == "__main__":

    # Read the successor features
    sfs = _get_successor_features("policies/Delivery-v0")
    
    # Instantiate the FSA
    fsa = fsa_delivery3()

    env = gym.make("DeliveryEval-v0")

    env = GridEnvWrapper(env, fsa)

    planning = ValueIteration(env.env, fsa, sfs)
   
    acc_reward = 0
    W = None
    times = [0]
    for _ in range(100):
        W, times_ = planning.traverse("u0", W, k=1)
        times.append(times_[-1])
        
        env.reset()
        acc_reward = 0

        for i in range(200):

            (f, state) = env.get_state()
            
            w = W[f]
            qvalues = np.asarray([np.dot(q[state], w) for q in sfs])

            action = np.unravel_index(qvalues.argmax(), qvalues.shape)
            obs, reward, done, phi = env.step(action[1])
            acc_reward+=reward

            # print(obs, phi["proposition"], acc_reward)

            if done:
                break
        
        print(acc_reward)
    
    print(np.cumsum(times).tolist())



    # print(np.round(V, 3))