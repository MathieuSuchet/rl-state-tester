from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward

from Rewards import TouchBallReward, SaveBoostReward, BoostPickupReward, PlayerVelocityReward

reward_functions = (
    TouchBallReward(),
    SaveBoostReward(),
    BoostPickupReward(),
    PlayerVelocityReward(),
    VelocityPlayerToBallReward(),
)

rewards_weight = (1, 1, 2, 0.002, 0.01)

reward_function = SB3CombinedLogReward(reward_functions, rewards_weight, "LogRewards")
