from typing import List, Union, Dict

import numpy as np

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.live_testing.player_recorder import PlayerAgent


class LivePlaying(Callback):
    def __init__(self, player_deadzone: float):
        self.player = PlayerAgent(player_deadzone)

    def _on_pre_step(self, actions: np.array):
        return np.array(self.player.get_controls())

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass