import gym
import numpy as np

gym.envs.register(
    id='DeepSeaTreasure-v0',
    entry_point='envs.dst:DeepSeaTreasure',
    max_episode_steps=100
)

gym.envs.register(
    id='ReacherMultiTask-v0',
    entry_point='envs.reacher:ReacherBulletEnv',
    max_episode_steps=100,
)

gym.envs.register(
    id='FourRoom-v0',
    entry_point='envs.four_room:Shapes',
    max_episode_steps=200
)


gym.envs.register(
    id='SimpleRoom-v0',
    entry_point='envs.room:Shapes',
    max_episode_steps=100
)

gym.envs.register(
    id='SimpleRoomNoisy-v0',
    entry_point='envs.room-noisy:Shapes',
    max_episode_steps=100
)

# Hallway
gym.envs.register(
    id='HallwaySingle-v0',
    entry_point='envs.hallway-single:Hallway',
    max_episode_steps=100,
    kwargs={'noise': 0}
)

gym.envs.register(
    id='HallwayNoisy-v1',
    entry_point='envs.hallway-single:Hallway',
    max_episode_steps=100,
    kwargs={'noise': 0.15}
)

gym.envs.register(
    id='HallwayNoisy-v2',
    entry_point='envs.hallway-single:Hallway',
    max_episode_steps=100,
    kwargs={'noise': 0.5}
)

gym.envs.register(
    id='HallwayMultiple-v0',
    entry_point='envs.hallway-multiple:Hallway',
    max_episode_steps=100,
    kwargs={'noise': 0}
)

gym.envs.register(
    id='HallwayNoisyMultiple-v1',
    entry_point='envs.hallway-multiple:Hallway',
    max_episode_steps=100,
    kwargs={'noise': 0.15}
)

gym.envs.register(
    id='HallwayNoisyMultiple-v2',
    entry_point='envs.hallway-multiple:Hallway',
    max_episode_steps=100,
    kwargs={'noise': 0.5}
)

gym.envs.register(
    id='CoffeeOffice-v0',
    entry_point='envs.coffee_office:CoffeeOffice',
    max_episode_steps=100,
)

gym.envs.register(
    id='CoffeeOffice-v1',
    entry_point='envs.coffee_office:CoffeeOffice',
    max_episode_steps=100,
    kwargs={'noise': 0.3}
)


gym.envs.register(
    id='CoffeeOffice-v2',
    entry_point='envs.coffee_office_learner:CoffeeOffice',
    max_episode_steps=200,
)



# HierarchicalOffice-v1

doorways = [((0, 0, 1, 2), (0, 1, 1, 0)), 
            ((0, 0, 2, 2), (1, 0, 3, 2))]

objects = {(0,0,0,0): "coffee1"}

initial_states = [(0, 0, 1, 1)]



gym.envs.register(
    id='HierarchicalOffice-v1',
    entry_point="envs.hierarchical_office_gridworld:HierarchicalOfficeGridworld",
    max_episode_steps=100,
    kwargs={

        "grid_size" : (2,2),
        "room_size": 5,
        "doorways": doorways,
        "initial_states": initial_states,
        "objects": objects

    }


)

gym.envs.register(
    id='Teleport-v0',
    entry_point='envs.teleport:Teleport',
    max_episode_steps=100,
)



gym.envs.register(
    id='Avoid-v0',
    entry_point='envs.avoid:Avoid',
    max_episode_steps=200,
)

gym.envs.register(
    id='Avoid-v1',
    entry_point='envs.avoid:Avoid',
    max_episode_steps=200,
    kwargs={'noise': 0.4}
    )