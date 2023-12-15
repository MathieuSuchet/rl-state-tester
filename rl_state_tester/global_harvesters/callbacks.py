from abc import ABC, abstractmethod
from typing import List, Union, Dict, Iterable

import numpy as np
from rlgym.api import AgentID, ActionType, RewardType, StateType, ObsType

from rl_state_tester.utils.commands.commands import Command


class Callback(ABC):
    def __init__(self, started_by_default: bool = False, commands: Iterable[Command] = None):
        self._started = started_by_default
        self.commands = commands
        if self.commands is None:
            self.commands = []


    def start(self):
        self._started = True

    @abstractmethod
    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        pass

    def on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        if self._started:
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
        if self._started:
            self._on_step(obs, action, reward, truncated, terminated, state, args, kwargs)

    @abstractmethod
    def _on_close(self, *args, **kwargs):
        pass

    def on_close(self, *args, **kwargs):
        if self._started:
            self._on_close(args, kwargs)

    @abstractmethod
    def _on_pre_step(self, actions: np.array, *args, **kwargs):
        pass

    def on_pre_step(self, actions: np.array, *args, **kwargs):
        return self._on_pre_step(actions, args, kwargs)


class MultiCallback(Callback):
    def __init__(self, callbacks: List[Callback], started_by_default: bool = True):
        super().__init__(started_by_default)
        self.callbacks = callbacks

    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_reset(obs, args, kwargs)

    def _on_step(self, obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        for callback in self.callbacks:
            callback.on_step(obs, action, reward, truncated, terminated, state, args, kwargs)

    def _on_close(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_close(args, kwargs)

    def _on_pre_step(self, actions: np.array, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_pre_step(actions)

