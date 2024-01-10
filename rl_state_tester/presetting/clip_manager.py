import os.path
from typing import List, Union, Dict, Tuple, Optional, Type

import numpy as np

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.presetting.clip_utils import load_clips, save_clips, DEFAULT_CLIP_PATH, DEFAULT_CLIP_LEGEND_PATH, \
    DEFAULT_STEPS_SAVED
from rl_state_tester.presetting.clipper import StateActionClipper
from rl_state_tester.presetting.recorder import ClipRecorder
from rl_state_tester.presetting.replayer import StateActionReplayer
from rl_state_tester.utils.commands import ClipManagerCommands
from rl_state_tester.utils.state_setters.common_setters import ClipSetter


class ClipManager(Callback):

    def __init__(self,
                 clip_path: str = DEFAULT_CLIP_PATH,
                 legend_clip_path: str = DEFAULT_CLIP_LEGEND_PATH,
                 clip_setter: ClipSetter = ClipSetter(),
                 n_steps_saved: int = DEFAULT_STEPS_SAVED,
                 commands: ClipManagerCommands = ClipManagerCommands(),
                 depends_on: Optional[List[Type]] = None):
        super().__init__(depends_on, commands)

        if not os.path.exists(os.path.join(os.getcwd(), clip_path)):
            open(os.path.join(os.getcwd(), clip_path), 'x').close()

        if not os.path.exists(os.path.join(os.getcwd(), legend_clip_path)):
            open(os.path.join(os.getcwd(), legend_clip_path), 'x').close()

        self.clip_setter = clip_setter
        self.clip_path = clip_path
        clips = load_clips(self.clip_path)

        commands.clipper_commands.clip.add_to_blocked(commands.replayer_commands.play_clip,
                                                      commands.unload_clips,
                                                      commands.save_clips, commands.load_clips)
        commands.replayer_commands.play_clip.add_to_blocked(commands.clipper_commands.clip)

        self.legend_clip_path = legend_clip_path
        self.last_observed = os.stat(os.path.join(os.getcwd(), clip_path)).st_mtime

        commands.unload_clips.target = self.del_clips
        commands.load_clips.target = self.load_clips
        commands.save_clips.target = self.save_clips

        self.active = False
        self.clipper = StateActionClipper(
            n_steps_saved=n_steps_saved,
            commands=commands.clipper_commands,
            clip_file_path=clip_path,
            legend_clip_file_path=self.legend_clip_path
        )
        self.replayer = StateActionReplayer(
            clips_file_path=clip_path,
            commands=commands.replayer_commands
        )

        self.recorder = ClipRecorder(
            clip_file_path=clip_path,
            clip_legend_file_path=legend_clip_path,
        )

        self.clipper.clips = clips if not isinstance(clips, type(None)) else []
        self.replayer.clips = clips if not isinstance(clips, type(None)) else []
        self.recorder.clips = clips if not isinstance(clips, type(None)) else []

    def start(self):
        super().start()
        self.clipper.start()
        self.replayer.start()
        self.recorder.start()

    def del_clips(self):
        if self._started:
            self.active = True

    def save_clips(self):
        if self._started:
            save_clips(self.clip_path, self.legend_clip_path, self.clipper.clips)

    def load_clips(self):
        if self._started:
            clips = load_clips(self.clip_path)
            self.clipper.clips = clips
            self.replayer.clips = clips
            self.recorder.clips = clips


    def update(self):
        if self._started:
            last_observed = self.last_observed = os.stat(os.path.join(os.getcwd(), self.clip_path)).st_mtime
            if last_observed != self.last_observed:
                self.load_clips()

    def _on_pre_step(self, actions: np.array):

        act = self.clipper.on_pre_step(actions)
        act = self.replayer.on_pre_step(act)
        act = self.recorder.on_pre_step(act)
        return act

    def _on_post_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                      terminal: Union[List[bool], bool],
                      info: Dict[str, object], *args, **kwargs) -> Tuple[List, List, bool, Dict]:
        o, r, t, i = self.clipper.on_post_step(obs, action, reward, terminal, info, args, kwargs)
        o, r, t, i = self.replayer.on_post_step(o, action, r, t, i, args, kwargs)
        o, r, t, i = self.recorder.on_post_step(o, action, r, t, i, args, kwargs)
        return o, r, t, i

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.clipper.on_reset(obs, info, args, kwargs)
        self.replayer.on_reset(obs, info, args, kwargs)
        self.recorder.on_reset(obs, info, args, kwargs)

    def _on_pre_reset(self):
        self.clipper.on_pre_reset()
        self.replayer.on_pre_reset()
        self.recorder.on_pre_reset()

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        self.clipper.on_step(obs, action, reward, terminal, info, args, kwargs)
        self.replayer.on_step(obs, action, reward, terminal, info, args, kwargs)
        self.recorder.on_step(obs, action, reward, terminal, info, args, kwargs)

        if self.active:
            choice = str(input("Are you sure you want to unload all the clips ? (y/n): "))
            if choice.lower() == 'y':
                self.clipper.clips = []
                self.replayer.reset()
            else:
                print("Aborted.")
            self.active = False

    def _on_close(self, *args, **kwargs):
        self.clipper.on_close(args, kwargs)
        self.replayer.on_close(args, kwargs)
        self.recorder.on_close(args, kwargs)

    def to_json(self):
        return {
            'clipper': self.clipper.to_json(),
            'replayer': self.replayer.to_json(),
            'recorder': self.replayer.to_json()
        }