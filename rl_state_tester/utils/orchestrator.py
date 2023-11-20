import time
from threading import Thread
from typing import List, NamedTuple, Type
import re

from rlgym.gym import Gym

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.commands import Command
from rl_state_tester.utils.envs import HarvestableEnv


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
                        c.target()
                time.sleep(self.command_detection_time)
        except Exception as e:
            print("Error on command detection thread:", e)


class Observer:
    def __init__(self, env: HarvestableEnv):
        self.env = env
        self.value = ''

    def update(self, value, *args):
        self.value = value
        if self.value == 'pause':
            self.env.toggle_pause()
        if self.value == 'reset':
            self.env.force_reset()
        if self.value == 'close':
            self.env.close()
        if self.value == 'change_ss_setter':
            self.env.update_ss_setter(*args)
        if self.value == 'add_clip':
            self.env.add_clip(*args)

        self.value = ''


class Distributor:
    class Link(NamedTuple):
        src: Callback
        dest: Type[Callback]

        def __str__(self):
            return f"{self.src.__class__.__name__} ---> {self.dest.__name__}"

    def __init__(self, harvesters: List[Callback], others: List):
        self.harvesters = harvesters
        self.others = others
        self.links = []

        self._distribute_dependencies()

    def _contains_element_of_type(self, t):
        for h in self.harvesters:
            if type(h) == t:
                return True
        for o in self.others:
            if type(o) == t:
                return True
        return False

    def _get_element_of_type(self, t):
        for h in self.harvesters:
            if type(h) == t:
                return h
        for o in self.others:
            if type(o) == t:
                return o

        return None


    def __verify_internal_dep(self, obj):
        for attr in obj.__dict__.values():
            if issubclass(type(attr), Callback):
                if not self._contains_element_of_type(type(attr)):
                    self.harvesters.append(attr)
                self.__verify_internal_dep(attr)
        if hasattr(obj, "dependencies"):
            deps = getattr(obj, "dependencies")
            for d in deps:
                if self._contains_element_of_type(d):
                    setattr(obj, Distributor.camel_to_snake(d.__name__), self._get_element_of_type(d))



    def _distribute_dependencies(self):
        for harvester in self.harvesters:

            self.__verify_internal_dep(harvester)

            if not hasattr(harvester, "dependencies"):
                continue

            for dep in harvester.dependencies:
                self.links.append(Distributor.Link(harvester, dep))

        self._create_instances()

    @staticmethod
    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def _has_link_from(self, origin):
        for l in self.links:
            if l.src == origin:
                return True
        return False

    def _create_instances(self):
        for link in self.links:

            elt_to_add = None

            for h in self.harvesters:
                if isinstance(h, link.dest):
                    elt_to_add = h
                    break

            for o in self.others:
                if isinstance(o, link.dest):
                    elt_to_add = o
                    break

            if elt_to_add is None:
                elt_to_add = link.dest()
                self.__verify_internal_dep(elt_to_add)
                if isinstance(elt_to_add, Callback):
                    self.harvesters.append(elt_to_add)

            setattr(link.src, Distributor.camel_to_snake(link.dest.__name__), elt_to_add)

        for h in self.harvesters:
            h.start()
