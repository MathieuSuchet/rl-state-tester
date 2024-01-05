from rl_state_tester.utils.commands.commands import LivePlayingCommands, Command
from rl_state_tester.utils.commands.commands_const import ACTIVATE_KEY
from rl_state_tester.utils.commands.configs import BindingConfig, Keys

DEFAULT_DEADZONE = 0.2
DEFAULT_COMMANDS = LivePlayingCommands(
    activate=Command(ACTIVATE_KEY, -1)
)

DEFAULT_CONFIG_JOYSTICK = BindingConfig(
    throttle=Keys.RIGHT_ANALOG,
    reverse_throttle=Keys.LEFT_ANALOG,

    handbrake=Keys.X_BUTTON,
    boost=Keys.B_BUTTON,
    jump=Keys.A_BUTTON,

    steer=Keys.LEFT_AXIS_X,
    yaw=Keys.LEFT_AXIS_Y,

    roll=Keys.X_BUTTON,
    air_roll_left=Keys.LEFT_TRIGGER,
    air_roll_right=Keys.RIGHT_TRIGGER
)
