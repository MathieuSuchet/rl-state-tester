from typing import Tuple, List, Union, Any, Dict

import numpy as np
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym as GymRL
from rlgym.utils import StateSetter
from rlgym_sim.gym import Gym as GymSim

from rl_state_tester.global_harvesters.callbacks import Callback, MultiCallback
from rl_state_tester.utils.rewards.reward_logger import RewardLogger


class HarvestableEnv(GymSim):

    def __init__(self, match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption, agent_tick_skip,
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
        obs, info = super().reset(True)
        self.harvester.on_reset(obs, info)

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

    def __init__(self, harvester: Callback, match, agent_tick_skip, launch_preference: LaunchPreference.EPIC, use_injector=False,
                 force_paging=False, raise_on_crash=False, auto_minimize=False):
        super().__init__(match, launch_preference=launch_preference, use_injector=use_injector, force_paging=force_paging, raise_on_crash=raise_on_crash, auto_minimize=auto_minimize)
        self.harvester = harvester
        self.agent_tick_skip = agent_tick_skip
        self.tick_skip = self._match.get_config()[2]
        self.current_tick = 0

    def reset(self, return_info=False) -> Union[List, Tuple]:
        obs, info = super().reset(True)
        self.harvester.on_reset(obs, info)

        if return_info:
            return obs, info
        return obs

    def step(self, actions: Any) -> Tuple[List, List, bool, Dict]:
        actions[0] = self.harvester.on_pre_step(actions)
        self.current_tick += self.tick_skip

        if self.current_tick < self.agent_tick_skip:
            for i in range(1, actions.shape[0]):
                actions[i] = np.zeros((1, 8))

        self.current_tick %= self.agent_tick_skip
        obs, reward, terminal, info = super().step(actions)

        self.harvester.on_step(obs, actions, reward, terminal, info)

        return obs, reward, terminal, info

    def close(self):
        super().close()
        self.harvester.on_close()
