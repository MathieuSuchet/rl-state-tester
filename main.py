import os

from rlgym.gamelaunch import LaunchPreference
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import reward_config
import rewards
import state_config
import states
from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.hot_reload.hot_reload import HotReload, HotReloadConfig
from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_sim, make_rl
from rl_state_tester.utils.rewards.reward_logger import RewardLogger

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


env = make_rl(
    tick_skip=1,
    agent_tick_skip=8,
    reward_fn=cb,
    state_setter=state_config.state_setter,
    team_size=3,
    launch_preference=LaunchPreference.STEAM,
    obs_builder=AdvancedObs(),
    action_parser=ContinuousAction(),
    spawn_opponents=True,
    terminal_conditions=[GoalScoredCondition(), TimeoutCondition(500)],
    harvester=MultiCallback(
        callbacks=[
            # Allow you to play as agent 0
            LivePlaying(player_deadzone=0.23),

            # Log rewards every %print_frequency% steps
            RewardLogger(
                reward_legends=[r.__class__.__name__ for r in cb.reward_functions],
                print_frequency=200),

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
            ))
        ])
)

agent = PPO(policy=MlpPolicy, env=env)

obs = env.reset()
running = True

hot_reload_path = "rewards.py"
last_modified_time = os.stat(hot_reload_path).st_mtime

while running:
    try:
        actions, _ = agent.predict(obs)

        obs, reward, terminal, info = env.step(actions)

        if terminal:
            obs = env.reset()

    except KeyboardInterrupt:
        print("Interruption detected, stopping")
        running = False

env.close()
