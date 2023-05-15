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
    max_episode_steps=100
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

gym.envs.register(
    id='Hallway-v0',
    entry_point='envs.hallway:Hallway',
    max_episode_steps=100
)

gym.envs.register(
    id='HallwayNoisy-v0',
    entry_point='envs.hallway-noisy:Hallway',
    max_episode_steps=100
)
