from typing import List, Union, Dict, Optional

import numpy as np
from rlgym_sim.gym import Gym
from stable_baselines3 import PPO

from rl_state_tester.global_harvesters.standalone_runner import StandaloneRunner


class RewardHarvester(StandaloneRunner):
    def __init__(
            self,
            env: Optional[Gym] = None,
            agent: Optional[PPO] = None,
            rendered: bool = False,
            deterministic: bool = True
    ):
        super().__init__(env, agent, rendered, deterministic)

        # All episodes shape   : [n_episodes, ?, n_players, 1] (Can't treat all episodes at once on axis 0 and 2 ?)
        # Episode Reward shape : [n_steps_in_ep, n_players, 1]
        # Reward shape         : [n_players, 1]
        self._all_rewards = [[]]
        self._n_episodes = -1

    def get_all_rewards(self):
        return [np.array(rewards) for rewards in self._all_rewards]

    @property
    def avg_current_episode_per_player(self):
        return np.mean(np.array(self._all_rewards[self._n_episodes]), axis=0)

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self._n_episodes += 1
        self._all_rewards.append([])

    def _on_step(self, obs: np.array, action, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        self._all_rewards[self._n_episodes].append(reward)

    def _on_close(self, *args, **kwargs):
        pass


class StateHarvester(StandaloneRunner):
    def __init__(self, env: Optional[Gym] = None, agent: Optional[PPO] = None, rendered: bool = False,
                 deterministic: bool = True):
        super().__init__(env, agent, rendered, deterministic)
        self._all_episodes = [[]]
        self._nb_episodes = -1

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.next_episode()

    def _on_step(self, obs: np.array, action, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        self.add_state_to_current_ep(state=info["state"])

    def _on_close(self, *args, **kwargs):
        pass

    def get_all_episodes(self):
        return [np.array(episode) for episode in self._all_episodes]

    def add_state_to_current_ep(self, state):
        self._all_episodes[self._nb_episodes].append(state)

    def get_states_for_episode(self, n_episode):
        return self._all_episodes[n_episode]

    def next_episode(self):
        self._nb_episodes += 1
        self._all_episodes.append([])


class ActionHarvester(StandaloneRunner):
    def __init__(self, env: Optional[Gym] = None, agent: Optional[PPO] = None, rendered: bool = False, deterministic: bool = True):
        super().__init__(env, agent, rendered, deterministic)
        self.actions = []
        self.n_episodes = -1

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.n_episodes += 1
        self.actions.append([])

    def _on_step(self, obs: np.array, action, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass