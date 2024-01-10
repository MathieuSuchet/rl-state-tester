from states.states import DefaultState, JumpShotState, SideHighRoll, WeightedVerifiedStateSetter

kickoff_prob = 0.1
airDribble2Touch_prob = 0
shotState_prob = 0.0
jumpShotState_prob = 0.1
airdribbleSetup_prob = 0
SideHighRoll_prob = 0.1
saveState_prob = 0.0

all_probs = [kickoff_prob, jumpShotState_prob, SideHighRoll_prob]
states = [DefaultState(), JumpShotState(), SideHighRoll()]

state_setter = WeightedVerifiedStateSetter(
    state_setters=states,
    weights=all_probs
)