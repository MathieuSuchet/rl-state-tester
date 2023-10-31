from typing import Tuple, Optional

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.reward_functions import CombinedReward


class SplitCombinedReward(CombinedReward):
    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None
    ):
        super().__init__(reward_functions, reward_weights)

    def get_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ):
        return [
            r.get_reward(player, state, previous_action) * float(w) for r, w in zip(self.reward_functions, self.reward_weights)
        ]

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ):
        return [
            r.get_final_reward(player, state, previous_action) * float(w) for r, w in zip(self.reward_functions, self.reward_weights)
        ]
