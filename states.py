from rlgym.utils import StateSetter
from rlgym.utils.state_setters import StateWrapper


class StateSetter1(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(10, 100, 100)


class StateSetter2(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(200, 25, 300)


class StateSetter3(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(456, 463, 186)


class StateSetter4(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.cars[0].set_pos(845, 465, 123)
