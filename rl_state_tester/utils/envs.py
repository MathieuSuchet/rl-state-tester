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
        engine_actions = self.action_parser.parse_actions(actions, self.state, self.shared_info)
        engine_actions = self.harvester.on_pre_step(engine_actions)
        new_state = self.transition_engine.step(engine_actions, self.shared_info)
        agents = self.agents
        obs = self.obs_builder.build_obs(agents, new_state, self.shared_info)
        is_terminated = self.termination_cond.is_done(agents, new_state, self.shared_info)
        is_truncated = self.truncation_cond.is_done(agents, new_state, self.shared_info)
        rewards = self.reward_fn.get_rewards(agents, new_state, is_terminated, is_truncated, self.shared_info)
        self.harvester.on_step(obs, actions, rewards, is_terminated, is_truncated, self.state)
        return obs, rewards, is_terminated, is_truncated

    def close(self) -> None:
        super().close()
        self.harvester.on_close()
