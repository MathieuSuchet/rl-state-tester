from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from states import StateSetter1, StateSetter2, StateSetter3, StateSetter4

state_setter = WeightedSampleSetter(
    state_setters=(StateSetter1(), StateSetter2(), StateSetter3(), StateSetter4()),
    weights=(1, 2, 3, 4)
)