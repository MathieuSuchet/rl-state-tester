from typing import Optional

from rlgym.api import StateMutator, ActionParser, AgentID, ActionType, EngineActionType, StateType, SpaceType, \
    ObsBuilder, ObsType, DoneCondition, RewardFunction, RewardType, TransitionEngine, Renderer
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.rocket_league.sim import RLViserRenderer, RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, MutatorSequence, FixedTeamSizeMutator
from rlgym_ppo.util import RLGymV2GymWrapper

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.commands.orchestrator import Orchestrator
from rl_state_tester.utils.envs import HarvestableEnv


def make(
        termination_cond: DoneCondition[AgentID, StateType] = NoTouchTimeoutCondition(timeout=500),
        truncation_cond: DoneCondition[AgentID, StateType] = NoTouchTimeoutCondition(timeout=500),
        reward_fn: RewardFunction[AgentID, StateType, RewardType] = CombinedReward(),
        obs_builder: ObsBuilder[AgentID, ObsType, StateType, SpaceType] = DefaultObs(),
        action_parser: ActionParser[AgentID, ActionType, EngineActionType, StateType, SpaceType] = LookupTableAction(),
        state_setter: StateMutator[StateType] = MutatorSequence(FixedTeamSizeMutator(blue_size=3,
                                                                                     orange_size=3), KickoffMutator()),
        transition_engine: TransitionEngine[AgentID, StateType, EngineActionType] = RocketSimEngine(),
        renderer: Optional[Renderer[StateType]] = RLViserRenderer(),
        callback: Optional[Callback] = None):
    """
    Returns an environment using a callback
    :param termination_cond: Termination conditions
    :param truncation_cond: Truncation conditions
    :param reward_fn: Reward function
    :param obs_builder: Observation builder
    :param action_parser: Action parser
    :param state_setter: State setter
    :param transition_engine: Transition engine (rocketsim / rocket league)
    :param renderer: Renderer (RLViser or nothing)
    :param callback: The callback to use
    :return: 
    """
    env = HarvestableEnv(
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_mutator=state_setter,
        reward_fn=reward_fn,
        termination_cond=termination_cond,
        callback=callback,
        renderer=renderer,
        transition_engine=transition_engine,
        truncation_cond=truncation_cond,
    )

    env.callback.start()

    Orchestrator(env.callback.commands.commands)

    return RLGymV2GymWrapper(env)
