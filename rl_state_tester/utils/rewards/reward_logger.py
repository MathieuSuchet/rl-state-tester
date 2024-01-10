import os
from typing import List, Union, Dict, Callable, Type, Optional

import numpy as np
from rlgym.utils.gamestates import GameState
from rlgym_sim.utils import RewardFunction as SimRewardFn
from rlgym_sim.utils.reward_functions import CombinedReward as SimCombinedReward

from rlgym.utils import RewardFunction as GymRewardFn
from rlgym.utils.reward_functions import CombinedReward as GymCombinedReward

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.rewards.common_rewards import RewardResult


def _print_rewards(state: GameState, rewards: List[Union[int, float, List]], rewards_legends: List[str]):
    os.system('cls')
    max_len = max([len(name) for name in rewards_legends])

    spacing = (max_len + 30)

    for i, player in enumerate(state.players):
        print("_" * spacing)
        print(f"|{'':<{spacing - 3}}", "|")
        print("| Player", player.car_id, ":" + f"{'|': >{spacing - (len(f'Player {player.car_id} :') + 2)}}")
        print(f"|{'':<{spacing - 3}}", "|")
        for r, l in zip(rewards[i], rewards_legends):
            if isinstance(r, RewardResult):
                print("| ", f"{l: <{max_len}}", ":",
                      f"{r.reward:<{10}.3f} {'(Error : ' + str(r.error) + ')' if r.error else ''}",
                      f"{'|': >{spacing - (max_len + 18)}}")
            else:
                print("| ", f"{l: <{max_len}}", ":",
                      f"{r:<{10}.3f} {'|': >{spacing - (max_len + 17)}}")

        print("|" + "_" * (spacing - 2) + "|\n")


class RewardLogger(Callback):

    def __init__(
            self,
            reward_function: Union[SimRewardFn, GymRewardFn],
            print_function: Callable[[GameState, List[Union[float, int, List]], List[str]], None] = _print_rewards,
            print_frequency: int = 10,
            depends_on: Optional[List[Type]] = None):
        super().__init__(depends_on)
        self.reward_fn = reward_function
        if issubclass(type(self.reward_fn), (SimCombinedReward, GymCombinedReward)) or isinstance(self.reward_fn, (SimCombinedReward, GymCombinedReward)):
            self.reward_legends = [c.__class__.__name__ for c in self.reward_fn.reward_functions]
        else:
            self.reward_legends = [self.reward_fn.__class__.__name__]
        self.print_func = print_function
        self.print_frequency = print_frequency
        self.count = 0

        self.last_actions = None

    def update_reward(self, reward, legends):
        if self._started:
            self.reward_fn = reward
            self.reward_legends = legends

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.last_actions = np.array([[0] * 8] * len(info["state"].players))

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):



        if self.count % self.print_frequency == 0 and self.count != 0:
            rewards = []
            for i, p in enumerate(info["state"].players):
                if issubclass(self.reward_fn.__class__, (SimCombinedReward, GymCombinedReward)) or isinstance(
                        self.reward_fn, (SimCombinedReward, GymCombinedReward)):
                    rewards.append([])
                    for reward_fn, reward_weight in zip(self.reward_fn.reward_functions, self.reward_fn.reward_weights):
                        try:
                            result = reward_fn.get_reward(p, info["state"], self.last_actions[i]) * reward_weight \
                                if not terminal \
                                else reward_fn.get_final_reward(p, info["state"], self.last_actions[i])
                            rewards[-1].append(result)
                        except Exception as e:
                            rewards[-1].append(RewardResult(0, e))
                else:
                    rewards.append(self.reward_fn.get_reward(p, info["state"], self.last_actions[i]) if not terminal
                                   else self.reward_fn.get_final_reward(p, info["state"], self.last_actions[i]))


            self.print_func(info["state"], rewards, self.reward_legends)
        self.count += 1
        self.last_actions = action

    def _on_close(self, *args, **kwargs):
        pass
