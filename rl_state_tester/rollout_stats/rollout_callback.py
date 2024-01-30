import multiprocessing as mp
import pickle
from typing import List, Union, Dict, Iterable, Type, Optional

import cloudpickle
import numpy as np
import rlgym
import rlgym_sim.make
import torch
from rlgym.gym import Gym as RLGym
from rlgym_ppo.ppo import PPOLearner
from rlgym_sim.gym import Gym as SimGym

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.rollout_stats.rollout_utils import RolloutBuffer, _process_target
from rl_state_tester.utils.commands import RolloutCommands
from rl_state_tester.utils.orchestrator import Observer

from rlgym_sim.utils.reward_functions import CombinedReward as SimCombinedReward
from rlgym.utils.reward_functions import CombinedReward as GymCombinedReward


class Rollout(Callback):
    def __init__(self, commands: RolloutCommands = RolloutCommands(), n_steps: int = 10_000, n_process: int = 1, sim: bool = True, tick_skip: int = 8,
                 depends_on: Optional[List[Type]] = None):
        if not depends_on:
            depends_on = [Observer]

        super().__init__(depends_on=depends_on, commands=commands)
        self.sim = sim
        self.n_process = n_process
        self.buffer = RolloutBuffer()
        self.n_steps = n_steps
        self.current_step = 0
        self.active = False
        self.tick_skip = tick_skip

        self.commands.start.target = self.activate
        self.commands.stop.target = self.deactivate

    def deactivate(self):
        self.active = False

    def __thread_target(self, n_process, env):
        agent = self.observer.update("get_agent")
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_process)])
        env_fns = [env for _ in range(n_process)]
        self.processes = []
        for work_remote, remote, env_fn in zip(
                self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, cloudpickle.dumps(env_fn))
            process = ctx.Process(
                target=_process_target, args=args, daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        def step(actions):
            flat_obs, flat_rew, flat_terminal, flat_info = [], [], [], []
            for i in range(n_process):
                self.remotes[i].send(("step", actions[i]))

            for i in range(n_process):
                obs, reward, terminal, info = self.remotes[i].recv()
                flat_obs.append(obs)
                flat_rew.append(reward)
                flat_terminal.append(terminal)
                flat_info.append(info)

            return \
                np.array(flat_obs), \
                    np.array(flat_rew), \
                    np.array(flat_terminal), \
                    np.array(flat_info)

        def reset():
            flat_obs, flat_info = [], []
            for i in range(n_process):
                self.remotes[i].send(("reset",))

            for i in range(n_process):
                obs, info = self.remotes[i].recv()
                flat_obs.append(obs)
                flat_info.append(info)

            return \
                np.array(flat_obs), \
                    np.array(flat_info)

        def close():
            for i in range(n_process):
                self.remotes[i].send(("close",))

        obs = reset()
        while self.current_step < self.n_steps or not self.active:
            actions = \
                np.array(agent.agent.policy.get_actions(obs)[0]) \
                if issubclass(agent.__class__, PPOLearner) \
                else agent.predict(obs)[0]

            actions = actions.reshape((*actions.shape, 1))
            obs, reward, terminal, info = step(actions)

            if any(terminal):
                obs = reset()

        close()
        self.active = False

    def activate(self):
        self.active = True
        env = self.observer.update("get_env")

        if self.sim and issubclass(env.__class__, RLGym):
            env = rlgym_sim.make(
                tick_skip=env._match._tick_skip,
                team_size=env._match._team_size,
                obs_builder=env._match._obs_builder,
                action_parser=env._match._action_parser,
                spawn_opponents=env._match._spawn_opponents,
                reward_fn=env._match._reward_fn,
                terminal_conditions=env._match._terminal_conditions,
                state_setter=env._match._state_setter
            )

        elif not self.sim and issubclass(env.__class__, SimGym):
            env = rlgym.make(
                tick_skip=self.tick_skip,
                team_size=env._match.team_size,
                obs_builder=env._match._obs_builder,
                action_parser=env._match._action_parser,
                spawn_opponents=env._match.spawn_opponents,
                reward_fn=env._match._reward_fn,
                terminal_conditions=env._match._terminal_conditions,
                state_setter=env._match._state_setter
            )

        def env_fn():
            return env

        self.__thread_target(self.n_process, env_fn)

    def __print_stats(self, rewards):
        reward_fn = self.observer.update("get_rewards")
        if issubclass(reward_fn.__class__, (SimCombinedReward, GymCombinedReward)) or isinstance(
                reward_fn, (SimCombinedReward, GymCombinedReward)):
            max_len = max([len(r.__class__.name) for r in reward_fn.reward_functions])
        else:
            max_len = len(reward_fn.__class__.__name__)
        spacing = (max_len + 30)
        print(f"{'Rollout logging':^-{spacing - 3}}")
        print(f"|{'':<{spacing - 3}}", "|")
        if issubclass(rewards.__class__, Iterable):
            for r, l in zip(rewards, self.reward_legends):
                print("| ", f"{l: <{max_len}}", ":",
                      f"{r:<{10}.3f} {'|': >{spacing - (max_len + 17)}}")
        print(f"|{'':<{spacing - 3}}", "|")

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        pass

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]],
                 terminal: Union[List[bool], bool], info: Dict[str, object], *args, **kwargs):
        pass

    def _on_close(self, *args, **kwargs):
        pass
