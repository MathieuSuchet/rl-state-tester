import numpy as np
from rlgym.gamelaunch import LaunchPreference

import reward_config
import rewards
import state_config
import states
from ArtemisParser import ArtemisParser
from AstraObs import AstraObs
from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.gymnasium_conversion.gymnasium_to_gym_conversion import convert_model_to_gym
from rl_state_tester.hot_reload.hot_reload import HotReload, HotReloadConfig
from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_sim, make_rl
from rl_state_tester.presetting.clip_manager import ClipManager
from rl_state_tester.utils.commands import ClipManagerCommands, StateActionClipperCommands, Command, \
    StateActionReplayerCommands, LivePlayingCommands
from rl_state_tester.utils.orchestrator import Orchestrator, Distributor, Observer
from rl_state_tester.utils.state_setters.common_setters import MultiSetter, ClipSetter

cb = reward_config.reward_function


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
    ),

    # Log rewards every %print_frequency% steps
    # RewardLogger(
    #     reward_legends=[r.__class__.__name__ for r in cb.reward_functions],
    #     print_frequency=200),

    # Hot reload of rewards and states
    HotReload(targets=(
        HotReloadConfig(
            script_path="rewards.py",
            config_path="reward_config.py",
            action=reward_action,
            script_module=rewards,
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
    reward_fn=cb,
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
    harvester=MultiCallback(
        callbacks=callbacks)
)
others = [
    Observer(env)
]

# Dependency injection I guess ?
distrib = Distributor(callbacks, others)

all_commands = []
for h in distrib.harvesters:
    all_commands.extend(h.commands.commands)

for h in distrib.others:
    if hasattr(h, "commands"):
        all_commands.extend(h.commands.commands)


orch = Orchestrator(
    commands=all_commands
)


# agent = convert_model_to_gym("rl_model_24403263860_steps.zip", env, 'cpu')

def create_env():
    return env


if __name__ == "__main__":
    # agent = Learner(
    #     env_create_function=create_env,
    #     ts_per_iteration=1000,
    #     timestep_limit=2000,
    #
    #     n_proc=1
    # )
    agent = convert_model_to_gym("rl_model_24403263860_steps(1).zip", env, 'cpu')
    obs, info = env.reset(True)
    running = True

    all_obs = [*obs[1:]]
    all_actions = []
    current_tick = 0

    while running:
        try:
            # actions, log_probs = agent.agent.policy.get_action(obs)
            # actions = actions.numpy().astype(np.float32)

            actions, _ = agent.predict(obs)
            actions = np.array(actions)

            if current_tick == 0:
                all_actions = actions[1:]
            else:
                actions[1:] = all_actions

            current_tick += 1
            current_tick %= 8

            obs, reward, terminal, info = env.step(actions)

            if terminal:
                obs, info = env.reset(True)

        except KeyboardInterrupt:
            print("Interruption detected, stopping")
            running = False

    env.close()
