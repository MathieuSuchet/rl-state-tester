from rlgym.gamelaunch import LaunchPreference
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward

import reward_config
import Rewards
import state_config
import states
from ArtemisParser import ArtemisParser
from AstraObs import AstraObs
from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.gymnasium_conversion.gymnasium_to_gym_conversion import convert_model_to_gym
from rl_state_tester.hot_reload.hot_reload import HotReload, HotReloadConfig
from rl_state_tester.init import run
from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_rl, make_sim
from rl_state_tester.presetting.clip_manager import ClipManager
from rl_state_tester.utils.commands import LivePlayingCommands
from rl_state_tester.utils.rewards.reward_logger import RewardLogger
from rl_state_tester.utils.state_setters.common_setters import MultiSetter, ClipSetter


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


clip_setter = ClipSetter(max_team_size=3)

callbacks = [
    # Allow you to play as agent 0
    LivePlaying(
        player_deadzone=0.23,
        commands=LivePlayingCommands()
    ),

    # Log rewards every %print_frequency% steps
    # RewardLogger(
    #     reward_legends=[r.__class__.__name__ for r in reward_config.reward_functions],
    #     print_frequency=200),

    # Hot reload of rewards and states
    HotReload(targets=(
        HotReloadConfig(
            script_path="rewards.py",
            config_path="reward_config.py",
            action=reward_action,
            script_module=Rewards,
            config_module=reward_config
        ),
        HotReloadConfig(
            script_path="states.py",
            config_path="state_config.py",
            action=state_action,
            script_module=states,
            config_module=state_config
        )
    )),
    ClipManager(
        clip_path='clips',
        legend_clip_path='clips_legend.txt',
        n_steps_saved=50,
        clip_setter=clip_setter,
    )
]

env = make_sim(
    tick_skip=1,
    # launch_preference=LaunchPreference.STEAM,
    reward_fn=ConstantReward(),
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
    terminal_conditions=[],
    harvester=MultiCallback(callbacks=callbacks)
)


if __name__ == "__main__":
    agent = convert_model_to_gym("rl_model_24403263860_steps.zip", env, "cpu")
    run(env, agent, callbacks, with_ui=True)
