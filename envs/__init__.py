import gym

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
    entry_point='envs.grid_envs:FourRooms',
    max_episode_steps=200
)


gym.envs.register(
    id='SimpleRoom-v0',
    entry_point='envs.grid_envs:Room',
    max_episode_steps=100
)

gym.envs.register(
    id='OfficeComplex-v0',
    entry_point='envs.grid_envs:OfficeComplex',
    max_episode_steps=100,
)

gym.envs.register(
    id='OfficeComplexEval-v0',
    entry_point='envs.grid_envs:OfficeComplex',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id='OfficeRS-v0',
    entry_point='envs.grid_envs:OfficeRS',
    max_episode_steps=100,
)

gym.envs.register(
    id='OfficeRSEval-v0',
    entry_point='envs.grid_envs:OfficeRS',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id='OfficeComplex-v1',
    entry_point='envs.grid_envs:OfficeComplex',
    max_episode_steps=100,
    kwargs={'random_act_prob': 0.3}
)


# Delivery
gym.envs.register(
    id='DeliveryMini-v0',
    entry_point='envs.grid_envs:DeliveryMini',
    max_episode_steps=200,
)

gym.envs.register(
    id='Delivery-v0',
    entry_point='envs.grid_envs:Delivery',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryMany-v0',
    entry_point='envs.grid_envs:DeliveryMany',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryEval-v0',
    entry_point='envs.grid_envs:Delivery',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
   id='DoubleSlit-v0',
   entry_point='envs.grid_envs:DoubleSlit',
   max_episode_steps=300,
)

gym.envs.register(
   id='DoubleSlitEval-v0',
   entry_point='envs.grid_envs:DoubleSlit',
   max_episode_steps=300,
)

gym.envs.register(
   id='DoubleSlitRS-v0',
   entry_point='envs.grid_envs:DoubleSlitRS',
   max_episode_steps=300,
)

# Teleport
gym.envs.register(
    id='Teleport-v0',
    entry_point='envs.grid_envs:Teleport',
    max_episode_steps=100,
)

gym.envs.register(
    id='IceCorridor-v0',
    entry_point='envs.grid_envs:IceCorridor',
    max_episode_steps=100,
)

