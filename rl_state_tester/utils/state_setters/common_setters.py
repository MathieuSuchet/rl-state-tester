import warnings
from typing import Type, Sequence, Optional

from rlgym.utils import StateSetter
from rlgym.utils.state_setters import StateWrapper

from rl_state_tester.presetting.clip_utils import Clip


class MultiSetter(StateSetter):
    def __init__(self, setters: Sequence[StateSetter]):
        self.setters = setters
        self.current_setter = self.setters[0]
        self.last_setter = None

    def add_clip(self, clip: Clip):
        if isinstance(self.current_setter, ClipSetter):
            self.current_setter.clip = clip

    def go_to(self, setter_type: Type[StateSetter], count: int = 0):
        current_count = 0
        found_setter = None
        for s in self.setters:
            if s.__class__ == setter_type:
                if current_count == count:
                    found_setter = s
                    break
                else:
                    current_count += 1

        if found_setter is None:
            warnings.warn(f"Can't go to setter {setter_type.__name__} number {count}")
        else:
            self.last_setter = self.current_setter
            self.current_setter = found_setter


    def reset(self, state_wrapper: StateWrapper):
        wrapper = self.current_setter.reset(state_wrapper)
        if isinstance(self.current_setter, ClipSetter):
            self.current_setter = self.last_setter if self.last_setter else self.current_setter

        return wrapper


class ClipSetter(StateSetter):
    def __init__(self, max_team_size: int = 4):
        self.max_team_size = max_team_size
        self.clip: Optional[Clip] = None

    def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
        if self.clip:
            return StateWrapper(game_state=self.clip.starting_state)
        return StateWrapper(blue_count=self.max_team_size, orange_count=self.max_team_size)

    def reset(self, state_wrapper: StateWrapper):
        if self.clip:
            starting_state = self.clip.starting_state

            # Ball setting
            ball = state_wrapper.ball

            ball.set_pos(*starting_state.ball.position)
            ball.set_lin_vel(*starting_state.ball.linear_velocity)
            ball.set_ang_vel(*starting_state.ball.angular_velocity)

            # Player settings
            for i, p in enumerate(state_wrapper.cars):
                state_player = starting_state.players[i]
                p.set_pos(*state_player.car_data.position)
                p.set_lin_vel(*state_player.car_data.linear_velocity)
                p.set_ang_vel(*state_player.car_data.angular_velocity)
                p.set_rot(state_player.car_data.pitch(), state_player.car_data.yaw(), state_player.car_data.roll())
                p.boost = state_player.boost_amount
                p.team_num = state_player.team_num
                p.id = state_player.car_id
