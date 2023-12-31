from abc import ABC, abstractmethod
from typing import List, Dict

from rlgym.api import AgentID, ActionType, RewardType, StateType, ObsType

from rl_state_tester.utils.commands.commands import Command, MultiCallbackCommands


class Callback(ABC):
    """
    The base class for all callbacks, allows you to interact with the environment on step, reset and close instruction.
    You can also interact with the actions sent to the environment
    """

    def __init__(self, started_by_default: bool = False, commands: Command = None):
        """
        :param started_by_default: True if the component can be active from the start, False if active at a certain moment (False by default)
        :param commands: Commands used by the component
        """
        self._started = started_by_default
        self.commands = commands

    def start(self):
        """
        Starts the component
        """
        self._started = True

    @abstractmethod
    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        """
        **You have to override this method.**\n
        Event that fires when the environment resets
        :param obs: The new obs received from the environment reset
        :param state: The state received from the environment reset
        """
        pass

    def on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        """
        Event that fires when the environment resets and if the component is active
        :param obs: The new observations received from the environment reset
        :param state: The state received from the environment reset
        :return:
        """
        if self._started:
            self._on_reset(obs, state, args, kwargs)

    @abstractmethod
    def _on_step(self,
                 obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        """
        **You have to override this method.**\n
        Event that fires when the environment steps
        :param obs: Observations received from the environment step
        :param action: Actions received from the environment step
        :param reward: Rewards received from the environment step
        :param truncated: Truncation flags received from the environment step
        :param terminated: Termination flags received from the environment step
        :param state: State received from the environment step
        """
        pass

    def on_step(self, obs: Dict[AgentID, ObsType],
                action: Dict[AgentID, ActionType],
                reward: Dict[AgentID, RewardType],
                truncated: Dict[AgentID, bool],
                terminated: Dict[AgentID, bool],
                state: StateType, *args, **kwargs):
        """
                Event that fires when the environment steps and the component is active
                :param obs: Observations received from the environment step
                :param action: Actions received from the environment step
                :param reward: Rewards received from the environment step
                :param truncated: Truncation flags received from the environment step
                :param terminated: Termination flags received from the environment step
                :param state: State received from the environment step
                """
        if self._started:
            self._on_step(obs, action, reward, truncated, terminated, state, args, kwargs)

    @abstractmethod
    def _on_close(self, *args, **kwargs):
        """
        **You have to override this method.**\n
        Event that fires when the environment closes
        """
        pass

    def on_close(self, *args, **kwargs):
        """
                **You have to override this method.**\n
                Event that fires when the environment closes and the component is active
                """
        if self._started:
            self._on_close(args, kwargs)

    @abstractmethod
    def _on_pre_step(self, actions: Dict[AgentID, ActionType], *args, **kwargs) -> Dict[AgentID, ActionType]:
        """
        **You have to override this method.**\n
        Event that fires before the environment step. Allows you to modify the actions sent to the environment
        :param actions: Actions that are about to be sent to the environment step
        :return: The new actions to step with
        """
        pass

    def on_pre_step(self, actions: Dict[AgentID, ActionType], *args, **kwargs) -> Dict[AgentID, ActionType]:
        """
        Event that fires before the environment step and if the component is active. Allows you to modify the actions sent to the environment
        :param actions: Actions that are about to be sent to the environment step
        :return: The new actions to step with
        """
        if self._started:
            return self._on_pre_step(actions, args, kwargs)
        return actions


class MultiCallback(Callback):
    """
    Allows you to have multiple callbacks
    """
    def __init__(self, callbacks: List[Callback], started_by_default: bool = True):
        """
        :param callbacks: The callbacks that'll be active during the environment loop. Keep in mind the callbacks will be called in the order you give them
        :param started_by_default: Same as base class Callback
        """
        super().__init__(started_by_default)
        self.callbacks = callbacks
        self.commands = MultiCallbackCommands()

        for callback in self.callbacks:
            self.commands.append_commands(callback.commands)

    def start(self):
        for callback in self.callbacks:
            callback.start()
        super().start()

    def _on_reset(self, obs: Dict[AgentID, ObsType], state: StateType, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_reset(obs, args, kwargs)

    def _on_step(self, obs: Dict[AgentID, ObsType],
                 action: Dict[AgentID, ActionType],
                 reward: Dict[AgentID, RewardType],
                 truncated: Dict[AgentID, bool],
                 terminated: Dict[AgentID, bool],
                 state: StateType,
                 *args, **kwargs):
        for callback in self.callbacks:
            callback.on_step(obs, action, reward, truncated, terminated, state, args, kwargs)

    def _on_close(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_close(args, kwargs)

    def _on_pre_step(self, actions: Dict[AgentID, ActionType], *args, **kwargs) -> Dict[AgentID, ActionType]:
        for callback in self.callbacks:
            actions = callback.on_pre_step(actions)
        return actions
