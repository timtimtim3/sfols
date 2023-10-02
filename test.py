import gym
import envs
from time import sleep
from rl.task_specifications import * 
from envs.wrapper import GridEnvWrapper
import random

if __name__ == '__main__':

    env = gym.make("Delivery-v0")
    print(env.exit_states)
    print([env.MAP[e] for e in env.exit_states])

    fsa = fsa_delivery1()

    env = GridEnvWrapper(env, fsa)
    env.reset()

