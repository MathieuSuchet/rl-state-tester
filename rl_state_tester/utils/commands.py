from typing import NamedTuple, Callable, List, Optional

import keyboard

from rl_state_tester.utils.commands_const import CLIP_KEY, PLAY_CLIP_KEY, LOAD_KEY, SAVE_KEY, \
    CLOSE_KEY, RESET_KEY, UNLOAD_KEY, ACTIVATE_KEY, PAUSE_KEY, STOP_CLIP_KEY, RECORD_KEY


class Hittable:
    @property
    def commands(self) -> List['Command']:
        return []

    def get_command(self, command: str):

        if hasattr(self, command):
            return self.__dict__[command]
        else:
            final_cmd = None
            for attr in self.__dict__.values():
                if issubclass(attr.__class__, Hittable):
                    final_cmd = attr.get_command(command)
                    if final_cmd:
                        break
            return final_cmd


class Command:
    def __init__(self, value: str, priority: int, target: Callable = lambda: None,
                 block: Optional[List['Command']] = None):
        self.value = value
        self.being_pressed = True
        self.priority = priority
        self.target = target
        self.blocked_commands = [] if isinstance(block, type(None)) else block
        self.blocked = False

    def add_to_blocked(self, *command):
        self.blocked_commands.extend(command)

    def is_pressed(self):
        keyboard_press = keyboard.is_pressed(self.value) and not keyboard.is_pressed('ctrl')

        if self.blocked:
            return False

        if not self.being_pressed and keyboard_press:
            self.being_pressed = True
            for c in self.blocked_commands:
                c.blocked = True
            return True

        if self.being_pressed:
            self.being_pressed = keyboard_press
            return False

        return False

    def unblock_all(self):
        for c in self.blocked_commands:
            c.blocked = False


class StateActionClipperCommands(Hittable):
    def __init__(self, clip: Command = Command(CLIP_KEY, -1)):
        self.clip = clip

    @property
    def commands(self):
        return [self.clip]


class StateActionReplayerCommands(Hittable):
    def __init__(
            self,
            play_clip: Command = Command(PLAY_CLIP_KEY, -1),
            stop_clip: Command = Command(STOP_CLIP_KEY, -1)
    ):
        self.play_clip = play_clip
        self.stop_clip = stop_clip

    @property
    def commands(self):
        return [self.play_clip, self.stop_clip]


class ClipRecorderCommands(Hittable):
    def __init__(
            self,
            toggle_recording: Command = Command(RECORD_KEY, -1)):
        self.toggle_recording = toggle_recording

    @property
    def commands(self):
        return [self.toggle_recording]


class MultiCallbackCommands(Hittable):
    def __init__(self):
        self._commands = []

    @property
    def commands(self) -> List['Command']:
        return self._commands

    def append_commands(self, command: Hittable):
        self._commands.extend(command.commands)


class ClipManagerCommands(Hittable):
    def __init__(
            self,
            clipper_commands: StateActionClipperCommands = StateActionClipperCommands(),
            replayer_commands: StateActionReplayerCommands = StateActionReplayerCommands(),
            unload_clips: Command = Command(UNLOAD_KEY, -1),
            load_clips: Command = Command(LOAD_KEY, -1),
            save_clips: Command = Command(SAVE_KEY, -1)
    ):
        self.clipper_commands = clipper_commands
        self.replayer_commands = replayer_commands
        self.unload_clips = unload_clips
        self.load_clips = load_clips
        self.save_clips = save_clips

    @property
    def commands(self):
        return [self.unload_clips, self.load_clips,
                self.save_clips]


class ForceCommands(Hittable):
    def __init__(
            self,
            reset: Command = Command(RESET_KEY, -1),
            close: Command = Command(CLOSE_KEY, -1),
            pause: Command = Command(PAUSE_KEY, -1)
    ):
        self.reset = reset
        self.close = close
        self.pause = pause

    @property
    def commands(self):
        return [self.reset, self.close, self.pause]


class LivePlayingCommands(Hittable):
    def __init__(self, activate: Command = Command(ACTIVATE_KEY, -1)):
        self.activate = activate

    @property
    def commands(self):
        return [self.activate]
