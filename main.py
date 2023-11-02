import os
from typing import Union, List

from rlgym.gamelaunch import LaunchPreference
from rlgym.utils.gamestates import GameState
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, ConstantReward, VelocityReward, \
    VelocityBallToGoalReward
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_rl, make_sim
from rl_state_tester.utils.rewards.RewardLogger import RewardLogger
from rl_state_tester.utils.rewards.common_rewards import SplitCombinedReward


class VeryVeryVeryLongRewardLabelThatNeedToBeShortened(VelocityReward):
    pass


reward_function = (
    EventReward(team_goal=120, goal=100, concede=-150, save=50, shot=60, demo=60),
    ConstantReward(),
    VelocityReward(),
    VelocityBallToGoalReward(),
    VeryVeryVeryLongRewardLabelThatNeedToBeShortened()
)

rewards_weight = (1, 15, 1, 8, 1)

cb = SplitCombinedReward(
    reward_functions=reward_function,
    reward_weights=rewards_weight
)


env = make_sim(
    tick_skip=1,
    reward_fn=cb,
    state_setter=DefaultState(),
    team_size=3,
    obs_builder=AdvancedObs(),
    action_parser=ContinuousAction(),
    spawn_opponents=True,
    terminal_conditions=[GoalScoredCondition()],
    harvester=MultiCallback(
        callbacks=[
            LivePlaying(player_deadzone=0.23),
            RewardLogger(
                reward_legends=[r.__class__.__name__ for r in reward_function],
                print_frequency=200)])
)

agent = PPO(policy=MlpPolicy, env=env)

obs = env.reset()
for i in range(500000):
    actions, _ = agent.predict(obs)

    obs, reward, terminal, info = env.step(actions)

    if terminal:
        obs = env.reset()

env.close()
