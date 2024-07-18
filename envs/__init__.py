import gym


gym.envs.register(
     id='ReacherMultiTask-v0',
     entry_point='envs.reacher:ReacherBulletEnv',
     max_episode_steps=100
)



# ANCHOR: Office environments
gym.envs.register(
    id='Office-v0',
    entry_point='envs.grid_envs:Office',
    max_episode_steps=200,
)

gym.envs.register(
    id='OfficeEval-v0',
    entry_point='envs.grid_envs:Office',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)


gym.envs.register(
    id='Office-v1',
    entry_point='envs.grid_envs:Office',
    max_episode_steps=200,
    kwargs={'random_act_prob': 0.3}
)

gym.envs.register(
    id='OfficeNegative-v0',
    entry_point='envs.grid_envs:OfficeNegative',
    max_episode_steps=200,
)

gym.envs.register(
    id='OfficeNegativeEval-v0',
    entry_point='envs.grid_envs:OfficeNegative',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)


# ANCHOR: Delivery environments
gym.envs.register(
    id='Delivery-v0',
    entry_point='envs.grid_envs:Delivery',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryEval-v0',
    entry_point='envs.grid_envs:Delivery',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryPenalty-v0',
    entry_point='envs.grid_envs:DeliveryPenalty',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryPenaltyEval-v0',
    entry_point='envs.grid_envs:DeliveryPenalty',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryMiniPenalty-v0',
    entry_point='envs.grid_envs:DeliveryMiniPenalty',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryMiniPenaltyEval-v0',
    entry_point='envs.grid_envs:DeliveryMiniPenalty',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

# ANCHOR: DoubleSlit environments
gym.envs.register(
    id='DoubleSlit-v0',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlitRS-v0',
    entry_point='envs.grid_envs:DoubleSlitRS',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlitRSEval-v0',
    entry_point='envs.grid_envs:DoubleSlitRS',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlit-v1',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={'max_wind': 3},
)

gym.envs.register(
    id='DoubleSlitEval-v0',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={
             'max_wind': 1},
)

gym.envs.register(
    id='DoubleSlitEval-v1',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={
             'max_wind': 3},
)

# ANCHOR: IceCorridor environments
gym.envs.register(
    id='IceCorridor-v0',
    entry_point='envs.grid_envs:IceCorridor',
    max_episode_steps=10000,
)

gym.envs.register(
    id='IceCorridorEval-v0',
    entry_point='envs.grid_envs:IceCorridor',
    max_episode_steps=10000,
)

gym.envs.register(
    id='PickupDropoff-v0',
    entry_point='envs.grid_envs:PickupDropoff',
    max_episode_steps=200,

)

gym.envs.register(
    id='PickupDropoffEval-v0',
    entry_point='envs.grid_envs:PickupDropoff',
    max_episode_steps=200,
    kwargs={'add_obj_to_start': False},

)