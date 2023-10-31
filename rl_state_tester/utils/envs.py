from typing import Tuple, List, Union, Any, Dict

from rlgym_sim.gym import Gym

from rl_state_tester.global_harvesters.callbacks import Callback


class HarvestableEnv(Gym):
    def __init__(self, match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption, harvester: Callback):
        super().__init__(match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption)
        self.harvester = harvester

    def reset(self, return_info=False) -> Union[List, Tuple]:
        obs, info = super().reset(True)
        self.harvester.on_reset(obs, info)

        if return_info:
            return obs, info
        return obs

    def step(self, actions: Any) -> Tuple[List, List, bool, Dict]:
        obs, reward, terminal, info = super().step(actions)

        self.harvester.on_step(obs, actions, reward, terminal, info)

        return obs, reward, terminal, info

    def close(self):
        super().close()
        self.harvester.on_close()