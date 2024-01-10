import copy
import json.tool
import time
from threading import Thread
from typing import List, Union, Dict, Optional, Type

import numpy as np
import pygame.joystick

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.live_testing.player_recorder import PlayerAgent
from rl_state_tester.utils.commands import LivePlayingCommands, Command
from rl_state_tester.utils.commands_const import ACTIVATE_KEY

DEFAULT_DEADZONE = 0.2
DEFAULT_COMMANDS = LivePlayingCommands(
    activate=Command(ACTIVATE_KEY, -1)
)


class LivePlaying(Callback):
    def __init__(
            self,
            player_deadzone: float = DEFAULT_DEADZONE,
            commands: LivePlayingCommands = DEFAULT_COMMANDS,
            active_by_default: bool = True,
            depends_on: Optional[List[Type]] = None
    ):
        super().__init__(depends_on, commands)
        self.player = None
        self.player_deadzone = player_deadzone
        self.active = active_by_default

        self.commands.activate.target = self.toggle

    def start(self):
        super().start()
        self.player = PlayerAgent(self.player_deadzone)
        self.player.start()

    def toggle(self):
        self.active = not self.active

    def _on_pre_step(self, actions: np.array):
        if self.active:
            act = copy.copy(actions)
            act[0] = np.array(self.player.get_controls())
            return act

        return actions

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass

    def to_json(self):
        base = super().to_json()
        base.update({
            'active': self.active
        })
        return base
