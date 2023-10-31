from typing import Dict

import numpy as np
from rlgym.api import AgentID, ObsType, ActionType, RewardType, StateType

from rl_state_tester.global_harvesters.callbacks import Callback


class RewardHarvester(Callback):
    def __init__(
            self
    ):

        # All episodes shape   : [n_episodes, ?, n_players, 1] (Can't treat all episodes at once on axis 0 and 2 ?)
        # Episode Reward shape : [n_steps_in_ep, n_players, 1]
        # Reward shape         : [n_players, 1]
        self._all_rewards = []
        self._n_episodes = -2

    def get_all_rewards(self):
        return [np.array(rewards) for rewards in self._all_rewards]

    @property
    def avg_current_episode_per_player(self):
        return np.mean(np.array(self._all_rewards[self._n_episodes]), axis=0)

    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        self._n_episodes += 1
        if self._n_episodes >= 0:
            self._all_rewards.append([])

    def _on_step(self,
                 obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        self._all_rewards[self._n_episodes].append(reward)

    def _on_close(self, *args, **kwargs):
        pass


class StateHarvester(Callback):
    def __init__(self):
        self._all_episodes = []
        self._nb_episodes = -2

    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        self.next_episode()

    def _on_step(self,
                 obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        self.add_state_to_current_ep(state=state)

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


class ActionHarvester(Callback):
    def __init__(self):
        self.actions = []
        self.n_episodes = -1

    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        self.n_episodes += 1
        if self.n_episodes >= 0:
            self.actions.append([])

    def _on_step(self,
                 obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        return action

    def _on_close(self, *args, **kwargs):
        pass