from rl_state_tester.utils.commands.commands import LivePlayingCommands, Command
from rl_state_tester.utils.commands.commands_const import ACTIVATE_KEY

DEFAULT_DEADZONE = 0.2
DEFAULT_COMMANDS = LivePlayingCommands(
    activate=Command(ACTIVATE_KEY, -1)
)