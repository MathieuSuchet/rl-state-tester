from typing import List, Union, Dict, Optional

import numpy as np
from rlgym_sim.gym import Gym
from stable_baselines3 import PPO

from global_harvesters.global_harvesters import RewardHarvester, StateHarvester
from standalone_runner import StandaloneRunner


class MyExampleHarvester(StandaloneRunner):
    def __init__(self, env: Optional[Gym] = None, agent: Optional[PPO] = None, rendered: bool = False,
                 deterministic: bool = True):
        super().__init__(env, agent, rendered, deterministic)

        # The harvesters you are using to get your data from
        self.my_harvesters = [
            StateHarvester(),
            RewardHarvester()
        ]

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):

        # Get your states
        my_states = self.my_harvesters[0].all_states

        # Get your rewards
        my_rewards = self.my_harvesters[1].all_rewards

        # For each episode
        for episode in range(my_states.shape[0]):
            for state, reward in zip(my_states[episode], my_rewards[episode]):
                # You can do things with your states and rewards
                # e.g.: Print player position with rewards
                for player in state.players:
                    print(player.position, ":", reward[0])

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass
