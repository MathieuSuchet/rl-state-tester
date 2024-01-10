import importlib
import os
from types import ModuleType
from typing import List, Union, Dict, Callable, Tuple, Optional, Type

import numpy as np

from rl_state_tester.global_harvesters.callbacks import Callback


class HotReloadConfig:

    def __init__(self, script_path: Optional[str], config_path: str, action: Callable, script_module: Optional[ModuleType], config_module: ModuleType):
        self.script_path = script_path
        self.config_path = config_path
        self.action = action
        self.script_module = script_module
        self.config_module = config_module
        self.last_modified = (
            os.stat(self.script_path).st_mtime,
            os.stat(self.config_path).st_mtime
        )

    def update(self):
        script_st_mtime = os.stat(self.script_path).st_mtime if self.script_path else 0
        config_st_mtime = os.stat(self.config_path).st_mtime

        updated = self.last_modified[0] != script_st_mtime or self.last_modified[1] != config_st_mtime

        self.last_modified = (
            script_st_mtime, config_st_mtime
        )
        return updated

    def reload(self):
        if self.script_module:
            importlib.reload(self.script_module)
        importlib.reload(self.config_module)



class HotReload(Callback):

    def __init__(
            self,
            targets: Optional[Tuple[HotReloadConfig, ...]] = None,
            depends_on: Optional[List[Type]] = None
    ):
        super().__init__(depends_on)
        self.targets = targets

    def _on_pre_step(self, actions: np.array):

        if isinstance(self.targets, type(None)):
            return actions

        for i, target in enumerate(self.targets):
            if target.update():
                try:
                    target.reload()
                    target.action()
                except Exception as e:
                    print("Problem during hot reload:", e)

        return actions

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass
