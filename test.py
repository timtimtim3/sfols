import gym
import envs
from time import sleep

if __name__ == '__main__':

    env = gym.make("DeliveryMini-v0")
    env.reset()

    print(env.unwrapped.initial)
    print(env.unwrapped.exit_states)

    for _ in range(20):
        print(env.step(env.action_space.sample()))
        env.render()
        sleep(1)
    sleep(10)
