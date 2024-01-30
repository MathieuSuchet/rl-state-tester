import os.path
from typing import Iterable

from rlgym.gamelaunch import LaunchPreference
from rlgym.utils import TerminalCondition as GymTerminalCondition
from rlgym_ppo import Learner
from rlgym_ppo.ppo import PPOLearner
from rlgym_sim.utils import TerminalCondition as SimTerminalCondition
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from stable_baselines3.ppo import PPO

import rewards.rewards
import rewards.reward_config
import states.states
from ArtemisParser import ArtemisParser
from AstraObs import AstraObs
from rewards import reward_config
from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.hot_reload.hot_reload import HotReload, HotReloadConfig
from rl_state_tester.init import run
from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_rl, make_sim
from rl_state_tester.rollout_stats.rollout_callback import Rollout
from rl_state_tester.utils.commands import LivePlayingCommands
from rl_state_tester.utils.rewards.reward_logger import RewardLogger
from rl_state_tester.utils.state_setters.common_setters import MultiSetter, ClipSetter
from states import state_config
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
    #     reward_function=reward_config.reward_function,
    #     print_frequency=100),

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
            script_module=states.states,
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
    Rollout(),
    #ClipManager(
    #    clip_path='clips',
    #    legend_clip_path='clips_legend.txt',
    #    n_steps_saved=50,
    #    clip_setter=clip_setter,
    #)
]

env = make_sim(
    tick_skip=1,
    agent_tick_skip=8,
    # launch_preference=LaunchPreference.STEAM,
    reward_fn=rewards.reward_config.reward_function,
    state_setter=MultiSetter(
        setters=[
            state_config.state_setter,
            clip_setter
        ]
    ),
    team_size=1,
    obs_builder=AdvancedObsPadder(),
    action_parser=ArtemisParser(),
    spawn_opponents=False,
    terminal_conditions=terminal_conditions_config.terminal_conditions,
    harvester=MultiCallback(callbacks=callbacks)
)


if __name__ == "__main__":
    # agent = PPO.load("exit_save.zip", env)
    agent = PPOLearner(
        device="cpu",
        act_space_size=env.action_space.n,
        obs_space_size=env.observation_space.shape[0],
        batch_size=100_000,
        clip_range=None,
        policy_type=0,
        continuous_var_range=None,
        critic_lr=1e-5,
        policy_lr=1e-5,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        n_epochs=10,
        ent_coef=.1,
        mini_batch_size=10_000,
    )

    # agent = convert_model_to_gym("exit_save.zip", env, "cpu")
    run(env, agent, callbacks, with_ui=False)
