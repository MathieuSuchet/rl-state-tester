import pygame.joystick

JUMP = 0
BOOST = 1
HANDBRAKE = 2

THROTTLE = 5
REVERSE = 4

STEER_AXIS = 0
YAW_AXIS = 1

AIR_ROLL_LEFT = 9
AIR_ROLL_RIGHT = 10

ROLL_BUTTON = 2


class PlayerAgent:
    def __init__(self, player_deadzone):
        pygame.joystick.init()
        pygame.init()
        self.stick = pygame.joystick.Joystick(0)
        self.stick.init()
        self.player_deadzone = player_deadzone
        self.last_jump = self.stick.get_button(JUMP)

        # car_controls.throttle = controls[i * n + 1]
        # car_controls.steer = controls[i * n + 2]
        # car_controls.pitch = controls[i * n + 3]
        # car_controls.yaw = controls[i * n + 4]
        # car_controls.roll = controls[i * n + 5]
        # car_controls.jump = controls[i * n + 6] == 1
        # car_controls.boost = controls[i * n + 7] == 1
        # car_controls.handbrake = controls[i * n + 8] == 1

    def get_controls(self):
        pygame.event.pump()
        jump = self.stick.get_button(JUMP)
        boost = self.stick.get_button(BOOST)
        handbrake = self.stick.get_button(HANDBRAKE)

        throttle = self.stick.get_axis(THROTTLE)
        throttle = max(0.0, throttle)

        reverse_throttle = self.stick.get_axis(REVERSE)
        reverse_throttle = max(0.0, reverse_throttle)

        throttle = throttle - reverse_throttle

        steer = self.stick.get_axis(STEER_AXIS)
        if abs(steer) < self.player_deadzone:
            steer = 0.

        pitch = self.stick.get_axis(YAW_AXIS)
        if abs(pitch) < self.player_deadzone:
            pitch = 0.

        yaw = steer

        roll = - self.stick.get_button(AIR_ROLL_LEFT) + self.stick.get_button(AIR_ROLL_RIGHT)
        roll_button = self.stick.get_button(ROLL_BUTTON)
        if roll_button or (jump and not self.last_jump):
            roll = steer
            yaw = 0
            
        self.last_jump = jump

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

