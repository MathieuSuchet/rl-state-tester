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
        self.stick = None
        self.player_deadzone = player_deadzone
        self.last_jump = None
        self.started = False

    def _check_for_joystick(self):
        try:
            self.stick = pygame.joystick.Joystick(0)
            return True
        except Exception:
            return False

    def start(self):
        try:
            self.stick = pygame.joystick.Joystick(0)
            self.stick.init()
            self.last_jump = self.stick.get_button(JUMP)
            self.started = True
        except Exception as e:
            print("Problem encountered when starting the live player:", e)

    def get_controls(self):
        if not self.started:
            return [0] * 8

        pygame.event.pump()

        try:
            self.stick.get_button(0)
        except Exception:
            if not self._check_for_joystick():
                return

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