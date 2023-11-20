import time
from threading import Thread
from typing import NamedTuple, List, Union, Dict, Tuple, Optional, Type

import numpy as np
from gym import Env

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.commands import ForceCommands
from rl_state_tester.utils.orchestrator import Observer


class ForceInstructor(Callback):
    def __init__(
            self,
            commands: ForceCommands = ForceCommands(),
            depends_on: Optional[List[Type]] = None):

        if depends_on is None:
            depends_on = [Observer]
        else:
            if Observer not in depends_on:
                depends_on.append(Observer)

        super().__init__(depends_on, commands)
        self.commands = commands
        self.running = True

        self.asking_for_reset = False
        self.asking_for_close = False
        self.asking_for_pause = False

        self.commands.reset.target = self.force_reset
        self.commands.close.target = self.force_close
        self.commands.pause.target = self.force_pause

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        if self.asking_for_close:
            self.asking_for_close = False
            self.observer.update('close')
        if self.asking_for_pause:
            # we wait
            while self.asking_for_pause:
                time.sleep(.1)

    def _on_post_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                      terminal: Union[List[bool], bool],
                      info: Dict[str, object], *args, **kwargs) -> Tuple[List, List, bool, Dict]:
        if self.asking_for_reset:
            self.asking_for_reset = False
            return obs, reward, True, info
        return obs, reward, terminal, info


    def _on_close(self, *args, **kwargs):
        pass

    def force_reset(self):
        self.asking_for_reset = True

    def force_close(self):
        self.asking_for_close = True

    def force_pause(self):
        print(f"{'Pausing' if not self.asking_for_pause else 'Unpausing'}")
        self.asking_for_pause = not self.asking_for_pause
