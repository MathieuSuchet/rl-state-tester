import copy
from typing import Dict

import numpy as np
from rlgym.api import AgentID, ObsType, StateType, ActionType, RewardType

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.live_testing.player_recorder import PlayerAgent
from rl_state_tester.utils.commands.commands import LivePlayingCommands, Command
from rl_state_tester.utils.commands.commands_const import ACTIVATE_KEY

DEFAULT_DEADZONE = 0.2
DEFAULT_COMMANDS = LivePlayingCommands(
    activate=Command(ACTIVATE_KEY, -1)
)


class LivePlaying(Callback):
    def __init__(
            self,
            commands: LivePlayingCommands,
            player_deadzone: float = DEFAULT_DEADZONE,
            active_by_default: bool = True,
    ):
        super().__init__()
        self.player = None
        self.player_deadzone = player_deadzone
        self.active = active_by_default
        self.starting = False
        self.commands = commands

        self.commands.activate.target = self.toggle

    def start(self):
        super().start()
        self.player = PlayerAgent(self.player_deadzone)

    def toggle(self):
        self.active = not self.active

    def _on_pre_step(self, actions: Dict[AgentID, ActionType], *args, **kwargs):
        if self.active:
            act = copy.copy(actions)
            act['blue-0'] = np.array(self.player.get_controls())
            return act
        return actions

    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        pass

    def _on_step(self, obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType, *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass