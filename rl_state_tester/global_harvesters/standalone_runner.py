import time
from abc import ABC
from typing import List, Union, Dict, Optional

import keyboard
import numpy as np
from rlgym_sim.gym import Gym
from stable_baselines3 import PPO

from rl_state_tester.global_harvesters.callbacks import Callback


class StandaloneRunner(Callback, ABC):
    def __init__(self, env: Optional[Gym] = None, agent: Optional[PPO] = None, rendered: bool = False, deterministic: bool = True):
        self._env = env
        self._agent = agent
        self._rendered = rendered
        self._n_steps = 0
        self._deterministic = deterministic
        # self.progress_bar = tqdm.tqdm(desc="Rollout progression")

    def run_standalone(self, target_steps):
        if not self._env:
            raise ValueError("You are trying to run in standalone mode without a valid environment (env is None)")


        obs, info = self._env.reset(return_info=True)
        obs = np.array(obs)
        self.on_reset(obs, info)

        if self._rendered:
            print("Press S for about a second to start the sample")
            while not keyboard.is_pressed("s"):
                time.sleep(.5)

        while self._n_steps < target_steps:
            if self._rendered:
                self._env.render()

            actions = np.array(self._agent.predict(obs, deterministic=True)[0])
            obs, reward, terminated, info = self._env.step(actions)
            obs = np.array(obs)

            if terminated:
                obs, info = self._env.reset(return_info=True)
                obs = np.array(obs)
                self.on_reset(obs, info)

            self.on_step(obs, actions, reward, terminated, info)

            if self._rendered:
                time.sleep(1 / (120 / 4))

            self._n_steps += obs.shape[0]
            # self.progress_bar.update(obs.shape[0])

        self._env.close()
        self.on_close()


class BaseRunner(StandaloneRunner):
    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.callback.on_reset(obs, info, args, kwargs)

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        self.callback.on_step(obs, action, reward, terminal, info, args, kwargs)

    def _on_close(self, *args, **kwargs):
        self.callback.on_close(args, kwargs)

    def __init__(self, env: Gym, agent: PPO, rendered: bool, deterministic: bool, callback: Callback):
        super().__init__(env, agent, rendered, deterministic)
        self.callback = callback
