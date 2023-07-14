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
    id='Coffee-v0',
    entry_point='envs.coffee:Coffee',
    max_episode_steps=100,
)

gym.envs.register(
    id='Coffee-v1',
    entry_point='envs.coffee:Coffee',
    max_episode_steps=100,
    kwargs={'noise': 0.3}
)


gym.envs.register(
    id='Office-v0',
    entry_point='envs.office:Office',
    max_episode_steps=100,
)

gym.envs.register(
    id='Office-v1',
    entry_point='envs.office:Office',
    max_episode_steps=100,
    kwargs={'noise': 0.3}
)

gym.envs.register(
    id='Teleport-v0',
    entry_point='envs.teleport:Teleport',
    max_episode_steps=100,
)
