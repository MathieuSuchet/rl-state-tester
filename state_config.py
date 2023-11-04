from states import StateSetter1, StateSetter2, StateSetter3, StateSetter4, WeightedVerifiedStateSetter

state_setter = WeightedVerifiedStateSetter(
    state_setters=(StateSetter1(), StateSetter2(), StateSetter3(), StateSetter4()),
    weights=(1, 2, 3, 4)
)