import os
import time
from typing import List, Union, Dict, Tuple, Optional, Type

import numpy as np
from rlgym.utils.gamestates import GameState
from rlgym.utils.state_setters import StateWrapper

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.global_harvesters.force_instructor import ForceInstructor
from rl_state_tester.presetting.clip_utils import load_clips, Clip
from rl_state_tester.utils.commands import StateActionReplayerCommands
from rl_state_tester.utils.orchestrator import Observer
from rl_state_tester.utils.state_setters.common_setters import ClipSetter


class StateActionReplayer(Callback):
    def __init__(
            self,
            clips_file_path,
            commands: StateActionReplayerCommands,
            depends_on=None):

        if depends_on is None:
            depends_on = [ForceInstructor, Observer]
        if Observer not in depends_on:
            depends_on.append(Observer)
        if ForceInstructor not in depends_on:
            depends_on.append(ForceInstructor)
        super().__init__(depends_on, commands)
        self.going_to_replay = False
        self.clips_file_path = clips_file_path
        self.clips = []

        self.last_modified_time = os.stat(self.clips_file_path).st_mtime
        self.active = False
        self.count = 0
        self.current_clip: Optional[Clip] = None
        self.focus = True

        self.commands.play_clip.target = self.activate_replaying

    def activate_replaying(self):
        if self._started:
            self.focus = True
            self.going_to_replay = True
            self.active = True

            while self.focus:
                time.sleep(.1)

    def _on_pre_reset(self):
        if len(self.clips) == 0 and self.active:
            print("No clip saved")
            self.active = False
            self.commands.play_clip.unblock_all()

        if self.active and self.current_clip is None:
            try:
                clip_choice = int(input("Which clip do you want to load ? (-1 to cancel): "))
                while clip_choice < -1 or clip_choice >= len(self.clips):
                    clip_choice = int(input(f"Clip number {clip_choice} doesn't exist, pick another one (-1 to cancel): "))
                    if clip_choice == -1:
                        self.active = False
                        self.current_clip = None
                        return

                self.current_clip = self.clips[clip_choice]
                print("Replaying...")
                self.observer.update('change_ss_setter', ClipSetter)
                self.observer.update('add_clip', self.current_clip)
                self.focus = False
            except Exception as e:
                print("Error when choosing the clip to replay:", e)

    def _on_pre_step(self, actions: np.array):
        if self.current_clip:
            return self.current_clip.actions[self.count]
        return actions

    def _on_post_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                      terminal: Union[List[bool], bool],
                      info: Dict[str, object], *args, **kwargs) -> Tuple[List, List, bool, Dict]:
        if self.going_to_replay:
            self.active = True
            self.going_to_replay = False
            return obs, reward, True, info
        return obs, reward, terminal, info

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        last_modified = os.stat(self.clips_file_path).st_mtime
        if last_modified != self.last_modified_time:
            self.clips = load_clips(self.clips_file_path)
            self.last_modified_time = last_modified

        if self.active and self.current_clip:
            if self.count == len(self.current_clip.actions) - 1:
                print("End of clip")
                self.active = False
                # self.force_instructor.force_pause()
                self.commands.play_clip.unblock_all()
                self.current_clip = None
                self.count = 0
            else:
                self.count += 1

    def _on_close(self, *args, **kwargs):
        pass

    def reset(self):
        if self._started:
            self.active = False
            self.clips = []
            self.current_clip = None
