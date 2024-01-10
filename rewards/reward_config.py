from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward

from rewards.rewards import TouchBallReward, SaveBoostReward, BoostPickupReward, PlayerVelocityReward, AerialReward

reward_functions = (
    TouchBallReward(),
    SaveBoostReward(),
    BoostPickupReward(),
    PlayerVelocityReward(),
    AerialReward()
)

rewards_weight = (1, 1, 2, 0.2, 0.04)

parameter_names = [
    "Touch Ball",
    "Save Boost",
    "Boost Pickup",
    "Player Velocity",
    "Aerial Reward",
]

reward_function = SB3CombinedLogReward(reward_functions, rewards_weight, "LogRewards")
