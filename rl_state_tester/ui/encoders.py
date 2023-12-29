import json
from json import JSONEncoder, JSONDecoder
from typing import Any

import numpy as np
from numpy import bool_
from rlgym.rocket_league.api import PhysicsObject, Car, GameConfig, GameState


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