from typing import List, Union, Dict

import numpy as np
from rlgym_sim.gym import Gym
from stable_baselines3 import PPO

from standalone_runner import StandaloneRunner


class StateHarvester(StandaloneRunner):
    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.next_episode()

    def _on_step(self, obs: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool], info: Dict[str, object],
                 *args, **kwargs):
        self.add_state_to_current_ep(state=info["state"])

    def _on_close(self, *args, **kwargs):
        pass

    def __init__(self, env: Gym, agent: PPO, rendered: bool, deterministic: bool):
        super().__init__(env, agent, rendered, deterministic)
        self._all_episodes = [[]]
        self._nb_episodes = -1

    def get_all_episodes(self):
        return [np.array(episode) for episode in self._all_episodes]

    def add_state_to_current_ep(self, state):
        self._all_episodes[self._nb_episodes].append(state)

    def get_states_for_episode(self, n_episode):
        return self._all_episodes[n_episode]

    def next_episode(self):
        self._nb_episodes += 1
        self._all_episodes.append([])
