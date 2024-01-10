from typing import List, Union, Dict, Optional, Type

import numpy as np

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.presetting.clip_utils import DEFAULT_STEPS_SAVED, DEFAULT_CLIP_PATH, \
    DEFAULT_CLIP_LEGEND_PATH, register_clips
from rl_state_tester.utils.commands import StateActionClipperCommands


class StateActionClipper(Callback):

    def __init__(
            self,
            n_steps_saved: int = DEFAULT_STEPS_SAVED,
            commands: StateActionClipperCommands = StateActionClipperCommands(),
            clip_file_path: str = DEFAULT_CLIP_PATH,
            legend_clip_file_path: str = DEFAULT_CLIP_LEGEND_PATH,
            depends_on: Optional[List[Type]] = None):
        super().__init__(depends_on, commands)
        self.n_steps_saved = n_steps_saved
        self.states = []
        self.actions = []
        self.clips = []
        self.clip_file_path = clip_file_path
        self.legend_clip_file_path = legend_clip_file_path

        self.running = True
        self.clipping = False

        self.commands.clip.target = self.activate_clipping

    def activate_clipping(self):
        if self._started:
            self.clipping = True

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        if self.clipping:
            register_clips(
                self.clips,
                state=self.states[0],
                actions=self.actions,
                clip_file_path=self.clip_file_path,
                legend_clip_file_path=self.legend_clip_file_path)
            self.clipping = False
            self.commands.clip.unblock_all()

        if len(self.actions) == self.n_steps_saved:
            self.states.pop(0)
            self.actions.pop(0)

        self.states.append(info['state'])
        self.actions.append(action)

    def reset(self):
        if self._started:
            self.clips = []
            self.states = []
            self.actions = []

    def _on_close(self, *args, **kwargs):
        pass
