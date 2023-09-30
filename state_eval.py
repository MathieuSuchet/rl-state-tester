from typing import List, Union, Dict

import numpy as np
from rlgym_sim.envs import Match
from rlgym_sim.gym import Gym
from rlgym_sim.utils import StateSetter
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityPlayerToBallReward
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from stable_baselines3 import PPO

from StateSetters import ProbabilisticStateSetter
from action_parsers import ArtemisParser
from callbacks import MultiCallback, Callback
from obs_builders import AstraObs
from reward_harvester import RewardHarvester
from reward_state_replayer import RewardStateReplayer
from rewards import GoalScoreSpeed, SaveBoostReward, BoostPickupReward, KickoffReward_MMR, WallReward, DribbleReward, \
    BumpReward, AerialReward, SplitCombinedReward
from standalone_runner import StandaloneRunner
from state_harvester import StateHarvester


class BaseRunner(StandaloneRunner):
    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.callback.on_reset(obs, info, args, kwargs)

    def _on_step(self, obs: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        self.callback.on_step(obs, reward, terminal, info, args, kwargs)

    def _on_close(self, *args, **kwargs):
        self.callback.on_close(args, kwargs)

    def __init__(self, env: Gym, agent: PPO, rendered: bool, deterministic: bool, callback: Callback):
        super().__init__(env, agent, rendered, deterministic)
        self.callback = callback


state_to_judge: StateSetter = ProbabilisticStateSetter()
team_size = 2
so = True
rendered = False

reward_function = (
    EventReward(team_goal=120, goal=100, concede=-150, save=50, shot=60, demo=60),
    GoalScoreSpeed(),
    SaveBoostReward(),
    BoostPickupReward(),
    KickoffReward_MMR(),
    WallReward(),
    DribbleReward(),
    BumpReward(),
    VelocityPlayerToBallReward(),
    AerialReward()
)

rewards_weight = (1, 10, 0.01, 2, 0.5, 0.000004, 0.0003, 1, 0.001, 0.00005)  # 0.0003,

combined_reward = SplitCombinedReward(
        reward_functions=reward_function,
        reward_weights=rewards_weight
    )

match = Match(
    spawn_opponents=so,
    team_size=team_size,
    terminal_conditions=[GoalScoredCondition(), TimeoutCondition(500)],
    action_parser=ArtemisParser(),
    state_setter=state_to_judge,
    obs_builder=AstraObs(),
    reward_function=combined_reward
)

env = Gym(match=match, boost_consumption=1, tick_skip=8, copy_gamestate_every_step=True, dodge_deadzone=0.8, gravity=1)

amount_of_steps = 500
rollout_steps = ((team_size * 2) if so else team_size) * amount_of_steps

model = PPO.load("models/exit_save.zip")

state_harvester = StateHarvester(env=env, agent=model, rendered=rendered, deterministic=True)
reward_harvester = RewardHarvester(env=env, agent=model, rendered=rendered, deterministic=True)

callbacks: MultiCallback = MultiCallback([
    state_harvester, reward_harvester
])


class SetSetter(StateSetter):
    def __init__(self, state: GameState):
        self.state = state

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            *self.state.ball.position
        )

        state_wrapper.ball.set_lin_vel(*self.state.ball.linear_velocity)
        state_wrapper.ball.set_ang_vel(*self.state.ball.angular_velocity)

        for i in range(len(state_wrapper.cars)):
            print(f"Player {i} : {self.state.players[i].car_data.position}")
            player = state_wrapper.cars[i]
            player.set_pos(*self.state.players[i].car_data.position)
            player.set_lin_vel(*self.state.players[i].car_data.linear_velocity)
            player.set_ang_vel(*self.state.players[i].car_data.angular_velocity)


if __name__ == "__main__":
    RewardStateReplayer(env, model, rendered, True, [re.__class__.__name__ for re in reward_function]).run_standalone(rollout_steps)

    # rewards = reward_harvester.get_all_rewards()
    # states = state_harvester.get_all_episodes()
    #
    # sum_rewards = np.sum(rewards[:, 0], axis=0)
    #
    # biggest_str = max([len(name) for name in [reward.__class__.__name__ for reward in reward_function]])
    #
    # print(f"Over {amount_of_steps} steps :\n")
    # for i, reward in enumerate(reward_function):
    #     print(f"{reward.__class__.__name__: <{biggest_str}} : {sum_rewards[i]:3f} "
    #           f"(Number of triggers : {combined_reward.triggers[i]})")

    # print(np.max(np.sum(rewards[:, 0], axis=0), axis=0))
    # max_reward_index = np.argmax(np.sum(rewards[:, 0], axis=0))
    # print(rewards[max_reward_index])
    # max_state: GameState = states[max_reward_index]
    #
    # match = Match(
    #     spawn_opponents=so,
    #     team_size=team_size,
    #     terminal_conditions=[GoalScoredCondition(), TimeoutCondition(500)],
    #     action_parser=ArtemisParser(),
    #     state_setter=SetSetter(max_state),
    #     obs_builder=AstraObs(),
    #     reward_function=CombinedReward(
    #         reward_functions=reward_function,
    #         reward_weights=rewards_weight
    #     )
    # )
    #
    # env = Gym(match=match, boost_consumption=1, tick_skip=8, copy_gamestate_every_step=True, dodge_deadzone=0.8, gravity=1)
    #
    # env.reset()
    # env.render()
    #
    # print(f"Player data : {max_state.players[0]}")
    #
    # print("Waiting for G press...")
    # while not keyboard.is_pressed("g"):
    #     time.sleep(0.5)
    #
    # try:
    #     while True:
    #         env.render()
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     env.close()
    #     sys.exit(1)
