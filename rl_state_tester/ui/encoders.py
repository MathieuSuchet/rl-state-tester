import json
import re
from json import JSONEncoder, JSONDecoder
from typing import Any

import numpy as np
from numpy import bool_
from rlgym.rocket_league.api import PhysicsObject, Car, GameConfig, GameState

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.commands import Hittable, Command


class JSONStateEncoder(JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, GameState):
            data = {}
            for s in o.__slots__:
                try:
                    attr = getattr(o, s)

                    if isinstance(attr, dict):
                        new_attr = {}
                        for k, v in attr.items():
                            if isinstance(v, Car):
                                new_attr.setdefault(k, JSONCarEncoder().default(v))
                        attr = new_attr

                    if isinstance(attr, bool_):
                        attr = bool(attr)

                    if isinstance(attr, Car):
                        attr = JSONCarEncoder().default(attr)

                    if isinstance(attr, PhysicsObject):
                        attr = JSONPhysicsObjectEncoder().default(attr)

                    if isinstance(attr, GameConfig):
                        attr = JSONGameConfigEncoder().default(attr)

                    if isinstance(attr, np.ndarray):
                        attr = attr.tolist()

                    data.setdefault(s, JSONDecoder().decode(super().encode(attr)))

                except Exception as e:
                    print(e)

            return data
        else:
            return json.dumps(o)


class JSONGameConfigEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        data = {}
        if isinstance(o, GameConfig):
            for s in o.__slots__:
                attr = getattr(o, s)
                data.setdefault(s, JSONDecoder().decode(super().encode(attr)))
            return data
        else:
            return json.dumps(o)


class JSONCarEncoder(JSONEncoder):
    def default(self, o: Any):
        data = {}
        if isinstance(o, Car):

            for s in o.__slots__:
                try:
                    attr = getattr(o, s)
                    if isinstance(attr, np.ndarray):
                        attr = attr.tolist()

                    if isinstance(attr, PhysicsObject):
                        attr = JSONPhysicsObjectEncoder().default(attr)

                    data.setdefault(s, JSONDecoder().decode(super().encode(attr)))
                except Exception as e:
                    print(e)
            return data
        else:
            return json.dumps(o)


class JSONPhysicsObjectEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, PhysicsObject):
            data = {}
            for s in o.__slots__:
                attr = getattr(o, s)
                if isinstance(attr, np.ndarray):
                    attr = attr.tolist()

                data.setdefault(s, JSONDecoder().decode(super().encode(attr)))
            return data
        else:
            return json.dumps(o)


class JSONCallbackInfoEncoder(JSONEncoder):
    def default(self, callback: Any):
        if isinstance(callback, Callback):
            class_name = callback.__class__.__name__
            split_name = re.findall('[A-Z0-9][^A-Z0-9]*', class_name)
            class_name = ""
            path = ""

            for elt in split_name:
                class_name += elt + " "
                path += elt.lower() + "-"
            class_name = class_name[:-1]
            path = path[:-1]

            return {
                'id': id(callback),
                'name': class_name,
                'path': path,
                'commands': JSONHitterEncoder().default(callback.commands),
                'subcommands': JSONHitterEncoder().default_sub(callback.commands),
                'callback': callback.to_json()
            }


class JSONCommandEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Command):
            return {
                'value': o.value
            }


class JSONHitterEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Hittable):
            names = []
            for k, v in o.__dict__.items():
                if issubclass(v.__class__, Hittable):
                    continue
                else:
                    names.append({
                        'name': k,
                        'key': JSONCommandEncoder().default(o.__dict__[k])
                    })
            return names
        else:
            return json.dumps(o)

    def default_sub(self, o: Any) -> Any:
        if isinstance(o, Hittable):
            names = []
            for k, v in o.__dict__.items():
                if issubclass(v.__class__, Hittable):
                    names.append({
                        'commands': self.default(v),
                        'component': v.__class__.__name__
                    })
            return names
