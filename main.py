from typing import Iterable, List, Optional

from rlgym.gamelaunch import LaunchPreference
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.utils import TerminalCondition as GymTerminalCondition
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_sim.utils import TerminalCondition as SimTerminalCondition
from stable_baselines3.ppo import PPO

import rewards.rewards
import rewards.reward_config
from ArtemisParser import ArtemisParser
from AstraObs import AstraObs
from rewards import reward_config
from rl_state_tester.global_harvesters.callbacks import MultiCallback, Callback
from rl_state_tester.global_harvesters.force_instructor import ForceInstructor
from rl_state_tester.hot_reload.hot_reload import HotReload, HotReloadConfig
from rl_state_tester.init import run
from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_rl, make_sim
from rl_state_tester.presetting.clip_manager import ClipManager
from rl_state_tester.presetting.clip_utils import load_clips
from rl_state_tester.utils.commands import LivePlayingCommands
from rl_state_tester.utils.rewards.reward_logger import RewardLogger
from rl_state_tester.utils.state_setters.common_setters import MultiSetter, ClipSetter
from states import state_config, states
from states.states import DefaultState
from terminal_conditions import terminal_conditions_config, terminal_conditions


def reward_action():
    global env
    print("Hot reload request received")

    env.update_reward(reward_config.reward_function)
    print("Hot reload done")


def state_action():
    global env
    print("Hot reload request received")
    env.update_state(state_config.state_setter)
    print("Hot reload done")


def terminal_cond_action():
    global env
    if not issubclass(terminal_conditions_config.terminal_conditions.__class__, Iterable):
        print(f"Hot reload cancelled, expecting an Iterable, got "
              f"{terminal_conditions_config.terminal_conditions.__class__.__name__}")
        return

    for t in terminal_conditions_config.terminal_conditions:
        if not issubclass(t.__class__, (GymTerminalCondition, SimTerminalCondition)):
            print("Hot reload cancelled, wrong type of terminal conditions")
            return
    print("Hot reload request received")
    env.update_terminal_cond(terminal_conditions_config.terminal_conditions)
    print("Hot reload done")


clip_setter = ClipSetter(max_team_size=3)

callbacks = [
    # Allow you to play as agent 0
    LivePlaying(
        player_deadzone=0.23,
        commands=LivePlayingCommands()
    ),

    # Log rewards every %print_frequency% steps
    # RewardLogger(
    #    reward_function=reward_config.reward_function,
    #    print_frequency=100),

    # Hot reload of rewards and states
    HotReload(targets=(
        HotReloadConfig(
            script_path="./rewards/rewards.py",
            config_path="./rewards/reward_config.py",
            action=reward_action,
            script_module=rewards.rewards,
            config_module=reward_config
        ),
        HotReloadConfig(
            script_path="./states/states.py",
            config_path="./states/state_config.py",
            action=state_action,
            script_module=states,
            config_module=state_config
        ),
        HotReloadConfig(
            script_path="./terminal_conditions/terminal_conditions.py",
            config_path="./terminal_conditions/terminal_conditions_config.py",
            action=terminal_cond_action,
            script_module=terminal_conditions,
            config_module=terminal_conditions_config
        )
    )),
    ClipManager(
        clip_path='clips',
        legend_clip_path='clips_legend.txt',
        n_steps_saved=500,
        clip_setter=clip_setter,
    ),
    ForceInstructor()
]


def make_env(tick_skip: int = 8,
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
             harvester: Optional[Callback] = None,
             sim: bool = True):
    if sim:
        env = make_sim(
            tick_skip=tick_skip,
            reward_fn=reward_fn,
            state_setter=state_setter,
            team_size=team_size,
            obs_builder=obs_builder,
            action_parser=action_parser,
            spawn_opponents=spawn_opponents,
            terminal_conditions=terminal_conditions,
            harvester=harvester,

            gravity=gravity,
            boost_consumption=boost_consumption,
            dodge_deadzone=dodge_deadzone,
            copy_gamestate_every_step=copy_gamestate_every_step
        )
    else:
        env = make_rl(
            tick_skip=tick_skip,
            launch_preference=LaunchPreference.STEAM,
            reward_fn=reward_fn,
            state_setter=state_setter,
            team_size=team_size,
            obs_builder=obs_builder,
            action_parser=action_parser,
            spawn_opponents=spawn_opponents,
            terminal_conditions=terminal_conditions,
            harvester=harvester
        )
    return env


env = make_env(
    tick_skip=1,
    reward_fn=rewards.reward_config.reward_function,
    state_setter=MultiSetter(
        setters=[
            state_config.state_setter,
            clip_setter
        ]
    ),
    team_size=2,
    obs_builder=AstraObs(),
    action_parser=ArtemisParser(),
    spawn_opponents=True,
    terminal_conditions=terminal_conditions_config.terminal_conditions,
    harvester=MultiCallback(callbacks=callbacks),
    sim=False
)

if __name__ == "__main__":
    agent = PPO.load("exit_save.zip", env)
    # # agent = convert_model_to_gym("exit_save.zip", env, "cpu")
    run(env, agent, callbacks, with_ui=False)

