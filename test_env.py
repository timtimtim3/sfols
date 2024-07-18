import gym, envs
from rl import task_specifications
from time import sleep
import numpy as np

if __name__ == "__main__":


    # fsa = task_specifications.load_fsa("DeliverypPenaltyEval-v0-task1")
   env = gym.make("PickupDropoff-v0")

   print(env.reset())

   s, r, done, info  = env.step(2)

   while info["proposition"] != ' ':
      s, r, done, info = env.step(2)
      print(s, info)
   print(env.step(2))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(3))
   print(env.step(0))
   print(env.step(0))
   print(env.step(0))
   print(env.step(0))
   print(env.step(2))
      