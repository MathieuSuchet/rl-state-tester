from typing import List, Union, Dict, Optional, Type

import numpy as np

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.global_harvesters.force_instructor import ForceInstructor
from rl_state_tester.presetting.clip_utils import DEFAULT_CLIP_PATH, DEFAULT_CLIP_LEGEND_PATH, register_clips
from rl_state_tester.utils.commands import ClipRecorderCommands
from rl_state_tester.utils.orchestrator import Observer


class ClipRecorder(Callback):
    def __init__(
            self,
            clip_file_path: str = DEFAULT_CLIP_PATH,
            clip_legend_file_path: str = DEFAULT_CLIP_LEGEND_PATH,
            commands: ClipRecorderCommands = ClipRecorderCommands(),
            depends_on: Optional[List[Type]] = None
    ):
        super().__init__(depends_on, commands)

        self.clip_path = clip_file_path
        self.legend_file_path = clip_legend_file_path
        self.starting_state = None
        self.actions = []
        self.active = False
        self.clips = []

        self.commands.toggle_recording.target = self._set_active


    def _set_active(self):
        if self._started:
            print(f"{'Started recording' if not self.active else 'Stopped recording'}")
            self.active = not self.active

            if not self.active:
                register_clips(self.clips, self.starting_state, self.actions, clip_file_path=self.clip_path, legend_clip_file_path=self.legend_file_path)
                self.starting_state = None
                self.actions = []

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        if self.active:
            if not self.starting_state:
                self.starting_state = info['state']
            else:
                self.actions.append(action)

    def _on_close(self, *args, **kwargs):
        pass