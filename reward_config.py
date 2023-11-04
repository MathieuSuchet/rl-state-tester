from rewards import Reward1, Reward2, Reward3, Reward4
from rl_state_tester.utils.rewards.common_rewards import SplitCombinedReward

reward_function = SplitCombinedReward(
    (Reward1(), Reward2(), Reward3(), Reward4(), ),
    (1, 2, 4, 1)
)