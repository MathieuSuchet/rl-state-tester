from typing import Tuple, List, Union, Any, Dict

import numpy as np
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym as GymRL
from rlgym.rocket_league.game.communication import Message
from rlgym.utils import StateSetter
from rlgym_sim.gym import Gym as GymSim

from rlgym.envs.match import Match as GymMatch
from rlgym_sim.envs.match import Match as SimMatch

from rl_state_tester.global_harvesters.callbacks import Callback, MultiCallback
from rl_state_tester.utils.rewards.reward_logger import RewardLogger
from states import StateResetResult


class HarvestableEnv(GymSim):

    def __init__(self, match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption,
                 agent_tick_skip,
                 harvester: Callback):
        super().__init__(match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption)
        self.harvester = harvester
        self.agent_tick_skip = agent_tick_skip

    def update_reward(self, reward):
        self._match._reward_fn = reward
        if isinstance(self.harvester, RewardLogger):
            self.harvester.update_reward_legends([r.__class__.__name__ for r in reward.reward_functions])
        if isinstance(self.harvester, MultiCallback):
            for c in self.harvester.callbacks:
                if isinstance(c, RewardLogger):
                    c.update_reward_legends([r.__class__.__name__ for r in reward.reward_functions])

    def update_state(self, setter: StateSetter):
        self._match._state_setter = setter

    def reset(self, return_info=False) -> Union[List, Tuple]:
        """
                The environment reset function. When called, this will reset the state of the environment and objects in the game.
                This should be called once when the environment is initialized, then every time the `done` flag from the `step()`
                function is `True`.
                """

        state_str, error_result = self._match.get_reset_state()
        state = self._game.reset(state_str)

        self._match.episode_reset(state)
        self._prev_state = state

        obs = self._match.build_observations(state)
        info = {
            'state': state,
            'result': self._match.get_result(state)
        }
        self.harvester.on_reset(obs, info)

        if error_result:
            print("Error on state setting:", error_result.error)

        if return_info:
            return obs, info
        return obs

    def step(self, actions: Any) -> Tuple[List, List, bool, Dict]:
        actions = self.harvester.on_pre_step(actions)
        obs, reward, terminal, info = super().step(actions)

        self.harvester.on_step(obs, actions, reward, terminal, info)

        return obs, reward, terminal, info

    def close(self):
        super().close()
        self.harvester.on_close()


class HarvestableEnvRL(GymRL):

    def render(self, mode="human"):
        pass

    def __init__(self, harvester: Callback, match, agent_tick_skip, launch_preference: LaunchPreference.EPIC,
                 use_injector=False,
                 force_paging=False, raise_on_crash=False, auto_minimize=False):
        super().__init__(match, launch_preference=launch_preference, use_injector=use_injector,
                         force_paging=force_paging, raise_on_crash=raise_on_crash, auto_minimize=auto_minimize)
        self.harvester = harvester
        self.agent_tick_skip = agent_tick_skip
        self.tick_skip = self._match.get_config()[2]
        self.current_tick = 0

    def reset(self, return_info=False) -> Union[List, Tuple]:
        """
                The environment reset function. When called, this will reset the state of the environment and objects in the game.
                This should be called once when the environment is initialized, then every time the `done` flag from the `step()`
                function is `True`.
                """

        state_str, error_result = self._match.get_reset_state()

        exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER,
                                                    body=state_str)
        if exception is not None:
            self._handle_exception()
            exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER,
                                                        body=state_str)
            if exception is not None:
                import sys
                print("!UNABLE TO RECOVER ROCKET LEAGUE!\nEXITING")
                sys.exit(-1)

        state = self._receive_state()
        self._match.episode_reset(state)
        self._prev_state = state

        if self._auto_minimize:
            self._minimize_game()  # After a successful episode, try to minimize the game

        obs = self._match.build_observations(state)
        info = {
            'state': state,
            'result': self._match.get_result(state)
        }
        self.harvester.on_reset(obs, info)

        if error_result:
            print("Error on state setting:", error_result.error)

        if return_info:
            return obs, info
        return obs

    def step(self, actions: Any) -> Tuple[List, List, bool, Dict]:
        self.current_tick += self.tick_skip

        if self.current_tick < self.agent_tick_skip:
            for i in range(1, actions.shape[0]):
                actions[i] = np.zeros((1, 8))

        self.current_tick %= self.agent_tick_skip
        """
                The step function will send the list of provided actions to the game, then advance the game forward by `tick_skip`
                physics ticks using that action. The game is then paused, and the current state is sent back to RLGym. This is
                decoded into a `GameState` object, which gets passed to the configuration objects to determine the rewards,
                next observation, and done signal.

                :param actions: An object containing actions, in the format specified by the `ActionParser`.
                :return: A tuple containing (obs, rewards, done, info)
                """

        actions = self._match.parse_actions(actions, self._prev_state)
        actions = self.harvester.on_pre_step(actions)
        actions_sent = self._send_actions(actions)

        received_state = self._receive_state()

        # If, for any reason, the state is not successfully received, we do not want to just crash the API.
        # This will simply pretend that the state did not change and advance as though nothing went wrong.
        if received_state is None:
            print("FAILED TO RECEIEVE STATE! FALLING TO", self._prev_state)
            state = self._prev_state
        else:
            state = received_state

        obs = self._match.build_observations(state)
        done = self._match.is_done(state) or received_state is None or not actions_sent
        reward = self._match.get_rewards(state, done)
        self._prev_state = state

        info = {
            'state': state,
            'result': self._match.get_result(state)
        }

        self.harvester.on_step(obs, actions, reward, done, info)

        return obs, reward, done, info

    def close(self):
        super().close()
        self.harvester.on_close()


class VerifiedSimMatch(SimMatch):
    def get_reset_state(self) -> Tuple[list, StateResetResult]:
        new_state = self._state_setter.build_wrapper(self.team_size, self.spawn_opponents)
        state_result = self._state_setter.reset(new_state)
        state = new_state.format_state()

        return state, state_result


class VerifiedGymMatch(GymMatch):
    def get_reset_state(self) -> Tuple[list, StateResetResult]:
        new_state = self._state_setter.build_wrapper(self._team_size, self._spawn_opponents)
        result = self._state_setter.reset(new_state)
        return new_state.format_state(), result
