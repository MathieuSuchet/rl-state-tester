import os
import time
from typing import List, Union, Dict, Optional

import keyboard
import numpy as np
import rlviser_py
from rlgym_sim.gym import Gym
from rlgym_sim.utils.gamestates import GameState
from stable_baselines3 import PPO

from reward_harvester import RewardHarvester
from standalone_runner import StandaloneRunner
from state_harvester import StateHarvester


class RewardStateReplayer(StandaloneRunner):

    def __init__(
            self,
            env: Gym,
            agent: PPO,
            rendered: bool,
            deterministic: bool,
            reward_legends: List[str],
            state_harvester: Optional[StateHarvester] = None,
            reward_harvester: Optional[RewardHarvester] = None,

    ):
        super().__init__(env, agent, rendered, deterministic)
        self.state_harvester = state_harvester
        self.reward_harvester = reward_harvester
        self.current_state_index = 0
        self.nb_episodes = 0
        self.legends = reward_legends
        self.max_len = max([len(name) for name in self.legends])
        self.playing = False

        if not self.state_harvester:
            self.state_harvester = StateHarvester(env, agent, rendered, deterministic)

        if not self.reward_harvester:
            self.reward_harvester = RewardHarvester(env, agent, rendered, deterministic)


    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.state_harvester.on_reset(obs, info, args, kwargs)
        self.reward_harvester.on_reset(obs, info, args, kwargs)

    def _on_step(self, obs: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        self.state_harvester.on_step(obs, reward, terminal, info, args, kwargs)
        self.reward_harvester.on_step(obs, reward, terminal, info, args, kwargs)

    def _on_close(self, *args, **kwargs):
        self.state_harvester.on_close(args, kwargs)
        self.reward_harvester.on_close(args, kwargs)
        self._start_rendering()

    def _start_rendering(self):
        states = self.state_harvester.get_all_episodes()
        rewards = self.reward_harvester.get_all_rewards()

        self.current_state_index = 0
        self.nb_episodes = 0

        rlviser_py.render_rlgym(states[self.nb_episodes][self.current_state_index])

        keyboard.on_press_key("right arrow", lambda e: self._step_forward(states, rewards))
        keyboard.on_press_key("left arrow", lambda e: self._step_backward(states, rewards))
        keyboard.on_press_key("p", lambda e: self._play_from_step(states, rewards))


        print("Waiting for shift...")
        keyboard.wait("shift")

    def _print_rewards(self, state: GameState, rewards):
        for i in range(len(state.players)):
            print(f"Player {i}:")
            for j, legend in enumerate(self.legends):
                print(f"\t{legend: <{self.max_len}} : {float(rewards[i][j]):3f}")

    def _step_forward(self, states, rewards):

        self.current_state_index += 1
        # Reset
        if states[self.nb_episodes].shape[0] <= self.current_state_index:
            if self.nb_episodes >= len(states):
                self.current_state_index -= 1
                if self.playing:
                    keyboard.press("s")
                return

            self.nb_episodes += 1
            self.current_state_index = 0
            print("Resetting to first state of next episode")

        rlviser_py.render_rlgym(states[self.nb_episodes][self.current_state_index])
        os.system("cls")
        self._print_rewards(states[self.nb_episodes][self.current_state_index], rewards[self.nb_episodes][self.current_state_index])
    def _step_backward(self, states, rewards):
        self.current_state_index -= 1
        # Reset
        if self.current_state_index < 0:
            if self.nb_episodes <= 0:
                self.current_state_index = 0
                return

            self.nb_episodes -= 1
            self.current_state_index = states[self.nb_episodes].shape[0] - 1
            print("Resetting to first state of next episode")

        rlviser_py.render_rlgym(states[self.nb_episodes][self.current_state_index])
        os.system("cls")
        self._print_rewards(states[self.nb_episodes][self.current_state_index], rewards[self.nb_episodes][self.current_state_index])

    def _play_from_step(self, states, rewards):
        if self.playing:
            return

        keyboard.on_press_key("s", lambda e: self._stop_playing())

        self.playing = True
        while self.playing:
            self._step_forward(states, rewards)
            time.sleep(1 / (120 / 4))

    def _stop_playing(self):
        self.playing = False
