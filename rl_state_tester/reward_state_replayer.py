import os
from typing import Dict, Optional

import keyboard
from rlgym.api import AgentID, ObsType, StateType, ActionType, RewardType
from rlviser_py import rlviser_py

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.global_harvesters.global_harvesters import StateHarvester, RewardHarvester
from rl_state_tester.utils.rewards.common_rewards import SplitCombinedReward


class RewardStateReplayer(Callback):

    def __init__(
            self,
            rendered: bool,
            combined_reward: SplitCombinedReward,
            state_harvester: Optional[StateHarvester] = None,
            reward_harvester: Optional[RewardHarvester] = None,

    ):
        self.state_harvester = state_harvester
        self.reward_harvester = reward_harvester
        self.current_state_index = 0
        self.nb_episodes = 0
        self.legends = [r.__class__.__name__ for r in combined_reward.reward_fns]
        self.max_len = max([len(name) for name in self.legends])
        self.playing = False
        self.rendered = rendered
        self.combined_reward = combined_reward

        if not self.state_harvester:
            self.state_harvester = StateHarvester()

        if not self.reward_harvester:
            self.reward_harvester = RewardHarvester()


    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        self.state_harvester.on_reset(obs, state, args, kwargs)
        self.reward_harvester.on_reset(obs, state, args, kwargs)

    def _on_step(self,
                 obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        self.state_harvester.on_step(obs, action, reward, terminated, truncated, state, args, kwargs)
        self.reward_harvester.on_step(obs, action, reward, terminated, truncated, state, args, kwargs)

    def _on_close(self, *args, **kwargs):
        self.state_harvester.on_close(args, kwargs)
        self.reward_harvester.on_close(args, kwargs)
        self._start_rendering()

    def _start_rendering(self):
        states = self.state_harvester.get_all_episodes()
        rewards = self.reward_harvester.get_all_rewards()

        self.current_state_index = 0
        self.nb_episodes = 0

        if self.rendered:
            rlviser_py.render_rlgym(states[self.nb_episodes][self.current_state_index])

        keyboard.on_press_key("right arrow", lambda e: self._step_forward(states, rewards))
        keyboard.on_press_key("left arrow", lambda e: self._step_backward(states, rewards))
        keyboard.on_press_key("p", lambda e: self._play_from_step(states, rewards))

        print("Waiting for shift...")
        keyboard.wait("shift")

    def _print_rewards(self, state: StateType, rewards):
        for i, name in enumerate(state.cars.keys()):
            print(f"Player {name}:")

            for j, legend in enumerate(self.legends):
                print("\t", legend, ":", rewards[name][j])
            #     step = self.combined_reward.steps[i][self.nb_episodes][self.current_state_index] if self.combined_reward.steps[i] is not None else None
            #     print(f"\t{legend: <{self.max_len}} : {float(rewards[i][j]):3f} "
            #           f"{((step.value if step.value < 0 else ('+' + step.value)) + ':' + step.reason if isinstance(self.combined_reward.steps[i], list) else 'Nothing') if step else ''}")

    def _step_forward(self, states, rewards):
        print("Forward")

        self.current_state_index += 1
        # Reset
        if states[self.nb_episodes].shape[0] <= self.current_state_index:
            if self.nb_episodes + 2 >= len(states):
                self.current_state_index -= 1
                if self.playing:
                    self.playing = False
                return

            self.nb_episodes += 1
            self.current_state_index = 0
            print("Resetting to first state of next episode")

        if self.rendered:
            rlviser_py.render_rlgym(states[self.nb_episodes][self.current_state_index])

        os.system("cls")
        self._print_rewards(
            states[self.nb_episodes][self.current_state_index],
            rewards[self.nb_episodes][self.current_state_index]
        )

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

        if self.rendered:
            rlviser_py.render_rlgym(states[self.nb_episodes][self.current_state_index])

        os.system("cls")
        self._print_rewards(states[self.nb_episodes][self.current_state_index], rewards[self.nb_episodes][self.current_state_index])

    def _play_from_step(self, states, rewards):
        if self.playing:
            return

        self.playing = True
        while self.playing:
            self._step_forward(states, rewards)

            if keyboard.is_pressed("s"):
                self.playing = False

    def _stop_playing(self):
        self.playing = False
