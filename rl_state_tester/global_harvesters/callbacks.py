from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional, Type, Tuple

import numpy as np

from rl_state_tester.utils.commands import Hittable
from rl_state_tester.utils.rewards.common_rewards import RewardResult


class Callback(ABC):
    def __init__(self, depends_on: Optional[List[Type]] = None, commands: Hittable = Hittable()):
        if not isinstance(depends_on, type(None)):
            # Remove duplicates
            self.dependencies = list(dict.fromkeys(depends_on))
        self._started = False
        self.commands = commands

    def start(self):
        self._started = True

    def on_post_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                     terminal: Union[List[bool], bool],
                     info: Dict[str, object], *args, **kwargs) -> Tuple[List, List, bool, Dict]:
        if self._started:
            return self._on_post_step(obs, action, reward, terminal, info, args, kwargs)
        return obs, reward, terminal, info

    def _on_post_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                      terminal: Union[List[bool], bool],
                      info: Dict[str, object], *args, **kwargs) -> Tuple[List, List, bool, Dict]:
        return obs, reward, terminal, info

    def _on_pre_step(self, actions: np.array):
        return actions

    def on_pre_step(self, actions: np.array):
        if self._started:
            return self._on_pre_step(actions)
        return actions

    @abstractmethod
    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        if self._started:
            self._on_reset(obs, info, args, kwargs)

    @abstractmethod
    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        pass

    def on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                terminal: Union[List[bool], bool],
                info: Dict[str, object], *args, **kwargs):
        if self._started:
            self._on_step(obs, action, reward, terminal, info, args, kwargs)

    @abstractmethod
    def _on_close(self, *args, **kwargs):
        pass

    def on_close(self, *args, **kwargs):
        if self._started:
            self._on_close(args, kwargs)

    def on_pre_reset(self):
        if self._started:
            self._on_pre_reset()

    def _on_pre_reset(self):
        pass

    def to_json(self):
        return {
            'id': id(self),
            'name': self.__class__.__name__
        }


class MultiCallback(Callback):
    def __init__(self, callbacks: List[Callback]):
        super().__init__()
        self.callbacks = callbacks

    def start(self):
        for callback in self.callbacks:
            callback.start()
        super().start()

    def _on_post_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                      terminal: Union[List[bool], bool],
                      info: Dict[str, object], *args, **kwargs) -> Tuple[List, List, bool, Dict]:
        for callback in self.callbacks:
            obs, reward, terminal, info = callback.on_post_step(obs, action, reward, terminal, info, args, kwargs)
        return obs, reward, terminal, info

    def _on_pre_reset(self):
        for callback in self.callbacks:
            callback.on_pre_reset()

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
