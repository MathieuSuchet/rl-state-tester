import os
from typing import List, Union, Dict, Callable, Type, Optional, Sequence

import numpy as np
from rlgym.utils.gamestates import GameState

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.utils.rewards.common_rewards import RewardResult


def _print_rewards(state: GameState, rewards: List[Union[int, float, List]], rewards_legends: List[str]):
    os.system('cls')
    max_len = max([len(name) for name in rewards_legends])

    spacing = (max_len + 30)

    for i, player in enumerate(state.players[:2]):
        print("_" * spacing)
        print(f"|{'':<{spacing - 3}}", "|")
        print("| Player", player.car_id, ":" + f"{'|': >{spacing - (len(f'Player {player.car_id} :') + 2)}}")
        print(f"|{'':<{spacing - 3}}", "|")
        for r, l in zip(rewards[i], rewards_legends):
            if isinstance(r, RewardResult):
                print("| ", f"{l: <{max_len}}", ":",
                      f"{r.reward:<{10}.3f} {'(Error : ' + str(r.error) + ')' if r.error else ''}",
                      f"{'|': >{spacing - (max_len + 18)}}")

        print("|" + "_" * (spacing - 2) + "|\n")


class RewardLogger(Callback):

    def __init__(
            self,
            reward_legends: Optional[Sequence[str]] = None,
            print_function: Callable[[GameState, List[Union[float, int, List]], List[str]], None] = _print_rewards,
            print_frequency: int = 10,
            depends_on: Optional[List[Type]] = None):
        super().__init__(depends_on)
        self.reward_legends = reward_legends
        if isinstance(reward_legends, type(None)):
            self.reward_legends = []
        self.print_func = print_function
        self.print_frequency = print_frequency
        self.count = 0

    def update_reward_legends(self, legends):
        if self._started:
            self.reward_legends = legends

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        if self.count % self.print_frequency == 0 and self.count != 0:
            self.print_func(info["state"], reward, self.reward_legends)
        self.count += 1

    def _on_close(self, *args, **kwargs):
        pass
