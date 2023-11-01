import os

from rlgym.gamelaunch import LaunchPreference
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, ConstantReward, VelocityReward, \
    VelocityBallToGoalReward
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from rl_state_tester.live_testing.live_playing import LivePlaying
from rl_state_tester.make import make_rl
from rl_state_tester.utils.rewards.common_rewards import SplitCombinedReward

reward_function = (
        EventReward(team_goal=120, goal=100, concede=-150, save=50, shot=60, demo=60),
        ConstantReward(),
        VelocityReward(),
        VelocityBallToGoalReward()
    )

rewards_weight = (1, 15, 1, 8)

cb = SplitCombinedReward(
        reward_functions=reward_function,
        reward_weights=rewards_weight
    )

env = make_rl(
    tick_skip=1,
    agent_tick_skip=8,
    reward_fn=cb,
    state_setter=DefaultState(),
    team_size=3,
    launch_preference=LaunchPreference.STEAM,
    obs_builder=AdvancedObs(),
    action_parser=ContinuousAction(),
    spawn_opponents=True,
    terminal_conditions=[GoalScoredCondition()],
    harvester=LivePlaying(player_deadzone=0.23)
)

agent = PPO(policy=MlpPolicy, env=env)

obs = env.reset()
for i in range(500000):
    actions, _ = agent.predict(obs)

    obs, reward, terminal, info = env.step(actions)

    if i % 20 == 0 and i != 0:
        os.system('cls')

        for j, name in enumerate([r.__class__.__name__ for r in reward_function]):
            print(name + ":", f"{reward[0][j]:.3f}")

    if terminal:
        obs = env.reset()

env.close()