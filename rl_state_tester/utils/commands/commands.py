from typing import List, Optional, Callable

import keyboard

from rl_state_tester.utils.commands.commands_const import ACTIVATE_KEY


class Hittable:
    @property
    def commands(self) -> List['Command']:
        return []


class Command(Hittable):
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

class LivePlayingCommands(Hittable):
    def __init__(self, activate: Command = Command(ACTIVATE_KEY, -1)):
        self.activate = activate

    @property
    def commands(self):
        return [self.activate]

class MultiCallbackCommands(Hittable):
    def __init__(self):
        self._commands = []
    @property
    def commands(self) -> List['Command']:
        return self._commands

    def append_commands(self, command: Command):
        self._commands.extend(command.commands)
