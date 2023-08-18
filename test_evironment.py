import envs
import gym 
from envs.rm import *

if __name__ == "__main__":

    LEFT, RIGHT = 0, 1,

    env = gym.make("HallwaySingle-v0")
    # prop_sym_dim = 4  # coffee1, coffee2, office1, office2 (Propositions symbols dim)
    # state_dim = 3  # no coffee, have coffee, in office     (Reward State )
    # initial_state = 0 
    # terminal_states = (2,)
    # transitions = {
    #     (0, (1, 0, 0, 0)): (1, 0),  # no coffee, coffee1 -> have coffee, 0
    #     (0, (0, 1, 0, 0)): (1, 0),  # no coffee, coffee2 -> have coffee, 0
    #     (1, (0, 0, 1, 0)): (2, 1),  # have coffee, office1 -> in office, 1
    #     (1, (0, 0, 0, 1)): (2, 1),  # have coffee, office1 -> in office, 1
    # }
    
    # rm = SimpleRewardMachine(
    #     state_dim=state_dim,
    #     terminal_states=terminal_states,
    #     initial_state=0,
    #     prop_sym_dim=prop_sym_dim,
    #     transitions=transitions
    # )
    # henv = HierarchicalEnv(
    #     env=env,
    #     rm=rm
    # )


    # print(env.observation_space)
    

    obs = env.reset()



    print(env.step(RIGHT))
    print(env.step(RIGHT))
    print(env.step(RIGHT))
    # print(env.step(RIGHT))
    # print(env.step(LEFT))