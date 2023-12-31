from typing import Tuple, Union, List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.reward_functions import CombinedReward


class SplitCombinedReward(CombinedReward):
    """
    Reward that returns all its reward functions as a list rather than a float (don't use it in a normal context)
    """
    def __init__(self, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]):
        super().__init__(*rewards_and_weights)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        combined_rewards = {agent: [] for agent in agents}
        for reward_fn, weight in zip(self.reward_fns, self.weights):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent, reward in rewards.items():
                combined_rewards[agent].append(reward * weight)

        return combined_rewards
