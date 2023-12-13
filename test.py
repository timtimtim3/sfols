import gym 
import envs 

if __name__ == "__main__":

    env = gym.make("Office-v0")

    print(len(env.MAP), len(env.MAP[0]) )
