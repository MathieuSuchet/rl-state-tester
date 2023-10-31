from abc import ABC, abstractmethod
from typing import List, Union, Dict

import numpy as np
from rlgym.api import AgentID, ActionType, RewardType, StateType, ObsType


class Callback(ABC):

    @abstractmethod
    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        pass

    def on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        self._on_reset(obs, state, args, kwargs)

    @abstractmethod
    def _on_step(self,
                 obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        pass

    def on_step(self, obs: Dict[AgentID, ObsType],
                action: Dict[AgentID, ActionType],
                reward: Dict[AgentID, RewardType],
                truncated: Dict[AgentID, bool],
                terminated: Dict[AgentID, bool],
                state: StateType, *args, **kwargs):
        self._on_step(obs, action, reward, truncated, terminated, state, args, kwargs)

    @abstractmethod
    def _on_close(self, *args, **kwargs):
        pass

    def on_close(self, *args, **kwargs):
        self._on_close(args, kwargs)


class MultiCallback(Callback):
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def _on_reset(self, obs: np.array, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_reset(obs, args, kwargs)

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        for callback in self.callbacks:
            callback.on_step(obs, action, reward, terminal, info, args, kwargs)

    def _on_close(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_close(args, kwargs)
