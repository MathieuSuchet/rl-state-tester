from typing import List, Union, Dict

import numpy as np
from rlgym_sim.gym import Gym
from stable_baselines3 import PPO

from callbacks import Callback
from standalone_runner import StandaloneRunner


class RewardHarvester(StandaloneRunner):
    def __init__(self, env: Gym, agent: PPO, rendered: bool, deterministic: bool):
        super().__init__(env, agent, rendered, deterministic)

        # All episodes shape   : [n_episodes, ?, n_players, 1] (Can't treat all episodes at once on axis 0 and 2 ?)
        # Episode Reward shape : [n_steps_in_ep, n_players, 1]
        # Reward shape         : [n_players, 1]
        self.all_rewards = [[]]
        self.n_episodes = -1

    def get_all_rewards(self):
        return [np.array(rewards) for rewards in self.all_rewards]


    @property
    def avg_current_episode_per_player(self):
        return np.mean(np.array(self.all_rewards[self.n_episodes]), axis=0)

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.n_episodes += 1
        self.all_rewards.append([])

    def _on_step(self, obs: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        self.all_rewards[self.n_episodes].append(reward)

    def _on_close(self, *args, **kwargs):
        pass