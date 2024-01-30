from typing import List, Optional

from rlgym.envs.match import Match as RLMatch
from rlgym.gamelaunch import LaunchPreference
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.reward_functions import DefaultReward
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions import common_conditions

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.envs import HarvestableEnv, HarvestableEnvRL, VerifiedSimMatch, VerifiedGymMatch


def make_sim(tick_skip: int = 1,
             agent_tick_skip: int = 8,
             spawn_opponents: bool = False,
             team_size: int = 1,
             gravity: float = 1,
             boost_consumption: float = 1,
             copy_gamestate_every_step=True,
             dodge_deadzone=0.8,
             terminal_conditions: List[object] = (
                     common_conditions.TimeoutCondition(225), common_conditions.GoalScoredCondition()),
             reward_fn: object = DefaultReward(),
             obs_builder: object = DefaultObs(),
             action_parser: object = DefaultAction(),
             state_setter: object = DefaultState(),
             harvester: Optional[Callback] = None):
    match = VerifiedSimMatch(
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter,
        reward_function=reward_fn,
        terminal_conditions=terminal_conditions,
        spawn_opponents=spawn_opponents,
        team_size=team_size,
    )

    harvester.start()

    return HarvestableEnv(
        match=match,
        agent_tick_skip=tick_skip,
        copy_gamestate_every_step=copy_gamestate_every_step,
        dodge_deadzone=dodge_deadzone,
        tick_skip=tick_skip,
        boost_consumption=boost_consumption,
        gravity=gravity,
        harvester=harvester
    )


def make_rl(
        tick_skip: int = 8,
        spawn_opponents: bool = False,
        team_size: int = 1,
        launch_preference: LaunchPreference = LaunchPreference.EPIC,
        use_injector: bool = False,
        force_paging: bool = False,
        auto_minimize: bool = False,
        raise_on_crash: bool = False,
        terminal_conditions: List[object] = (
                common_conditions.TimeoutCondition(225), common_conditions.GoalScoredCondition()),
        reward_fn: object = DefaultReward(),
        obs_builder: object = DefaultObs(),
        action_parser: object = DefaultAction(),
        state_setter: object = DefaultState(),
        harvester: Optional[Callback] = None):
    match = VerifiedGymMatch(
        tick_skip=tick_skip,
        game_speed=1,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter,
        reward_function=reward_fn,
        terminal_conditions=terminal_conditions,
        spawn_opponents=spawn_opponents,
        team_size=team_size
    )

    harvester.start()

    return HarvestableEnvRL(
        match=match,
        launch_preference=launch_preference,
        use_injector=use_injector,
        force_paging=force_paging,
        auto_minimize=auto_minimize,
        raise_on_crash=raise_on_crash,
        harvester=harvester
    )
