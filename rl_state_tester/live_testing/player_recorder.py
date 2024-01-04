from rl_state_tester.live_testing.controls import ControlType, KeyboardRecorder, JoystickRecorder


class PlayerAgent:
    def __init__(self, path_to_config: str, player_deadzone, device: ControlType = ControlType.KEYBOARD):
        self.device = device
        self.control = None

        if self.device == ControlType.KEYBOARD:
            self.control = KeyboardRecorder()
        elif self.device == ControlType.JOYSTICK:
            self.control = JoystickRecorder(path_to_config, player_deadzone)

        self.player_deadzone = player_deadzone
        self.last_jump = None
        self.started = False

    def start(self):
        try:
            self.started = True
        except Exception as e:
            print("Problem encountered when starting the live player:", e)

    def get_controls(self):
        if not self.started:
            return [0] * 8

