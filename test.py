import gym
import envs
from time import sleep
from rl.task_specifications import * 
from envs.wrapper import GridEnvWrapper
import random
from time import sleep
import numpy as np

if __name__ == '__main__':

    env = gym.make("ReacherMultiTask-v0")
    env.render()
    env.reset()

    for _ in range(20000):
        s, r, done, info = env.step(env.action_space.sample())
        print(np.any(info['phi'] < 0), info['phi'])
        env.render('human')
        sleep(0.3)
