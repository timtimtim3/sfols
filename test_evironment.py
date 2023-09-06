import envs
import gym 
from envs.rm import *

if __name__ == "__main__":

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    env = gym.make("CoffeeOffice-v0")
    
    env.reset()
    env.unwrapped.reset((0,0))

    print(type(env))

    print(env.step(UP))
    print(env.step(UP))
    print(env.step(UP))
    print(env.step(UP))
    print(env.step(UP))
    print(env.step(UP))
    print(env.step(RIGHT))
    print(env.step(RIGHT))
    print(env.step(RIGHT))
    print(env.step(RIGHT))


    # print(env.step(RIGHT))
    # print(env.step(LEFT))