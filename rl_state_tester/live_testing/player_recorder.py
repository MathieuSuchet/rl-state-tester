import pygame.joystick


class PlayerAgent:
    def __init__(self, player_deadzone):
        pygame.joystick.init()
        pygame.init()
        self.stick = pygame.joystick.Joystick(0)
        self.stick.init()
        self.player_deadzone = player_deadzone

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
        jump = self.stick.get_button(0)
        boost = self.stick.get_button(1)
        handbrake = self.stick.get_button(2)

        throttle = self.stick.get_axis(5)
        throttle = max(0.0, throttle)

        reverse_throttle = self.stick.get_axis(4)
        reverse_throttle = max(0.0, reverse_throttle)

        throttle = throttle - reverse_throttle

        steer = self.stick.get_axis(0)
        if abs(steer) < self.player_deadzone:
            steer = 0.

        pitch = self.stick.get_axis(1)
        if abs(pitch) < self.player_deadzone:
            pitch = 0.

        yaw = steer

        roll = - self.stick.get_button(9) + self.stick.get_button(10)
        roll_button = self.stick.get_button(2)
        if roll_button or jump:
            roll = steer

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

