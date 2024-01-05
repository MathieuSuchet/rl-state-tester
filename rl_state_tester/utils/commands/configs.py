from enum import Enum


class Keys(Enum):
    A_BUTTON: int = 0
    B_BUTTON: int = 1
    X_BUTTON: int = 2
    Y_BUTTON: int = 3

    LEFT_AXIS: int = 4
    RIGHT_AXIS: int = 5

    LEFT_TRIGGER: int = 6
    LEFT_ANALOG: int = 7

    LEFT_DIRECTIONAL: int = 8
    RIGHT_DIRECTIONAL: int = 9
    DOWN_DIRECTIONAL: int = 10
    UP_DIRECTIONAL: int = 11

    RIGHT_TRIGGER: int = 12
    RIGHT_ANALOG: int = 13

    LEFT_AXIS_BUTTON: int = 14
    RIGHT_AXIS_BUTTON: int = 15

    RIGHT_AXIS_DOWN: int = 16
    RIGHT_AXIS_UP: int = 17
    RIGHT_AXIS_RIGHT: int = 18
    RIGHT_AXIS_LEFT: int = 19

    LEFT_AXIS_DOWN: int = 20
    LEFT_AXIS_UP: int = 21
    LEFT_AXIS_RIGHT: int = 22
    LEFT_AXIS_LEFT: int = 23

    LEFT_AXIS_X: int = 20
    LEFT_AXIS_Y: int = 21
    RIGHT_AXIS_X: int = 22
    RIGHT_AXIS_Y: int = 23


class BindingConfig:
    def __init__(self,
                 throttle: Keys,
                 reverse_throttle: Keys,
                 jump: Keys,
                 boost: Keys,
                 handbrake: Keys,
                 steer: Keys,
                 yaw: Keys,
                 roll: Keys,
                 air_roll_right: Keys,
                 air_roll_left: Keys
                 ):
        self.throttle = throttle
        self.reverse_throttle = reverse_throttle
        self.jump = jump
        self.boost = boost
        self.handbrake = handbrake
        self.steer = steer
        self.yaw = yaw
        self.roll = roll
        self.air_roll_right = air_roll_right
        self.air_roll_left = air_roll_left
