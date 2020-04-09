from gym.envs.registration import register

register(
    id='trader-v0',
    entry_point='gym_trader:TraderEnv',
)
