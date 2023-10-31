import numpy as np
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, ConstantReward, VelocityReward, \
    VelocityBallToGoalReward
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from rl_state_tester.make import make
from rl_state_tester.reward_state_replayer import RewardStateReplayer
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

env = make(
    tick_skip=8,
    reward_fn=cb,
    state_setter=DefaultState(),
    gravity=1,
    team_size=3,
    dodge_deadzone=0.8,
    boost_consumption=1,
    obs_builder=AdvancedObs(),
    action_parser=DiscreteAction(),
    spawn_opponents=True,
    terminal_conditions=[TimeoutCondition(500)],
    harvester=RewardStateReplayer(cb)
)

agent = PPO(policy=MlpPolicy, env=env)

obs = env.reset()
for i in range(500):
    actions, _ = agent.predict(obs)

    obs, reward, terminal, info = env.step(actions)

    if terminal:
        obs = env.reset()

env.close()