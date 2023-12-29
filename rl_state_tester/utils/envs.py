from typing import Tuple, Dict, Optional

from rlgym.api import RLGym, AgentID, ObsType, ActionType, RewardType, StateType, Renderer, TransitionEngine, \
    DoneCondition, RewardFunction, ActionParser, ObsBuilder, StateMutator, EngineActionType, SpaceType

from rl_state_tester.global_harvesters.callbacks import Callback


class HarvestableEnv(RLGym):
    def __init__(self,
                 state_mutator: StateMutator[StateType],
                 obs_builder: ObsBuilder[AgentID, ObsType, StateType, SpaceType],
                 action_parser: ActionParser[AgentID, ActionType, EngineActionType, StateType, SpaceType],
                 reward_fn: RewardFunction[AgentID, StateType, RewardType],
                 termination_cond: DoneCondition[AgentID, StateType],
                 truncation_cond: DoneCondition[AgentID, StateType],
                 transition_engine: TransitionEngine[AgentID, StateType, EngineActionType],
                 renderer: Optional[Renderer[StateType]],
                 harvester: Callback):
        super().__init__(state_mutator, obs_builder, action_parser, reward_fn, termination_cond, truncation_cond,
                         transition_engine, renderer)
        self.harvester = harvester

    def reset(self) -> Dict[AgentID, ObsType]:
        args = super().reset()
        self.harvester.on_reset(args, self.state)
        return args

    def step(self, actions: Dict[AgentID, ActionType]) -> Tuple[
            Dict[AgentID, ObsType], Dict[AgentID, RewardType], Dict[AgentID, bool], Dict[AgentID, bool]]:
        obs, rewards, terminated, truncated = super().step(actions)
        self.harvester.on_step(obs, actions, rewards, terminated, truncated, self.state)
        return obs, rewards, terminated, truncated

    def close(self) -> None:
        super().close()
        self.harvester.on_close()
