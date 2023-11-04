from typing import NamedTuple, Optional

from rlgym.utils import StateSetter
from rlgym.utils.gamestates import GameState
from rlgym.utils.state_setters import StateWrapper
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter


class StateResetResult(NamedTuple):
    state: StateWrapper
    error: Optional[Exception]


class WeightedVerifiedStateSetter(WeightedSampleSetter):

    def reset(self, state_wrapper: StateWrapper):
        try:
            super().reset(state_wrapper)
            return StateResetResult(state_wrapper, None)
        except Exception as e:
            return StateResetResult(state_wrapper, e)


class StateSetter1(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(100, 100, 100)
        raise Exception("Bad state setting, probably madex")


class StateSetter2(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(200, 25, 300)
        raise Exception("Bad state setting, probably madex")


class StateSetter3(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(456, 463, 186)
        raise Exception("Bad state setting, probably madex")


class StateSetter4(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(845, 465, 123)
        raise Exception("Bad state setting, probably madex")
