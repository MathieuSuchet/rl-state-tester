from typing import Tuple, Optional, NamedTuple, Union

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.reward_functions import CombinedReward


class RewardResult(NamedTuple):
    reward: Union[int, float]
    error: Optional[Exception]


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

        rewards = []
        for r, w in zip(self.reward_functions, self.reward_weights):
            try:
                rewards.append(RewardResult(r.get_reward(player, state, previous_action) * float(w), None))
            except Exception as e:
                rewards.append(RewardResult(0, e))

        return rewards

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ):
        rewards = []
        for r, w in zip(self.reward_functions, self.reward_weights):
            try:
                rewards.append(RewardResult(r.get_final_reward(player, state, previous_action) * float(w), None))
            except Exception as e:
                rewards.append(RewardResult(0, e))

        return rewards
