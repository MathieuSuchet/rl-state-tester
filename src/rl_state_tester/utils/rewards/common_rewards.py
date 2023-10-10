from typing import Tuple, Optional

from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.reward_functions import CombinedReward


class SplitCombinedReward(CombinedReward):
    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None
    ):
        super().__init__(reward_functions, reward_weights)
        self.steps = []

        # TODO: Implement steps logic
