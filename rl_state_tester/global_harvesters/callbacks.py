from abc import ABC, abstractmethod
from typing import List, Union, Dict

import numpy as np

from rl_state_tester.utils.rewards.common_rewards import RewardResult


class Callback(ABC):
    def _on_pre_step(self, actions: np.array):
        return actions

    def on_pre_step(self, actions: np.array):
        return self._on_pre_step(actions)

    @abstractmethod
    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self._on_reset(obs, info, args, kwargs)

    @abstractmethod
    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        pass

    def on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                terminal: Union[List[bool], bool],
                info: Dict[str, object], *args, **kwargs):
        self._on_step(obs, action, reward, terminal, info, args, kwargs)

    @abstractmethod
    def _on_close(self, *args, **kwargs):
        pass

    def on_close(self, *args, **kwargs):
        self._on_close(args, kwargs)


class MultiCallback(Callback):
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def _on_pre_step(self, actions: np.array):
        for callback in self.callbacks:
            actions = callback.on_pre_step(actions)
        return actions

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        for callback in self.callbacks:
            callback.on_reset(obs, info, args, kwargs)

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int, RewardResult]],
                 terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        for callback in self.callbacks:
            callback.on_step(obs, action, reward, terminal, info, args, kwargs)

    def _on_close(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_close(args, kwargs)
