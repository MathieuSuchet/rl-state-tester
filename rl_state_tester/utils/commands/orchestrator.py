import time
from threading import Thread
from typing import List

from rl_state_tester.utils.commands.commands import Command


class Orchestrator:
    def __init__(self, commands: List[Command], command_detection_time: float = 1e-2):
        self.commands = commands
        self.running = True
        self.command_detection_time = command_detection_time

        self.thread = Thread(target=self._detection_thread_activity)
        self.thread.start()

    @property
    def sorted_commands(self):
        return list(reversed(sorted(self.commands, key=lambda c: c.priority)))

    def _detection_thread_activity(self):
        commands: List[Command] = self.sorted_commands

        try:
            while self.running:
                for c in commands:
                    if c.is_pressed():
                        print(c.value, "has been pressed")
                        c.target()
                time.sleep(self.command_detection_time)
        except Exception as e:
            print("Error on command detection thread:", e)
