import sys
from abc import ABC, abstractmethod
from configparser import RawConfigParser, DuplicateSectionError, SectionProxy, MissingSectionHeaderError, \
    DuplicateOptionError
from typing import Optional

import pygame

from rl_state_tester.live_testing.player_consts import DEFAULT_DEADZONE


class ConfigParserMultiOpt(RawConfigParser):
    """ConfigParser allowing duplicate keys. Values are stored in a list"""

    def __init__(self):
        RawConfigParser.__init__(self, empty_lines_in_values=False, strict=False)

    def _read(self, fp, fpname):
        """Parse a sectioned configuration file.

    Each section in a configuration file contains a header, indicated by
    a name in square brackets (`[]'), plus key/value options, indicated by
    `name' and `value' delimited with a specific substring (`=' or `:' by
    default).

    Values can span multiple lines, as long as they are indented deeper
    than the first line of the value. Depending on the parser's mode, blank
    lines may be treated as parts of multiline values or ignored.

    Configuration files may include comments, prefixed by specific
    characters (`#' and `;' by default). Comments may appear on their own
    in an otherwise empty line or may be entered in lines holding values or
    section names.
    """
        elements_added = set()
        cursect = None  # None, or a dictionary
        sectname = None
        optname = None
        lineno = 0
        indent_level = 0
        e = None  # None, or an exception
        for lineno, line in enumerate(fp, start=1):
            comment_start = None
            # strip inline comments
            for prefix in self._inline_comment_prefixes:
                index = line.find(prefix)
                if index == 0 or (index > 0 and line[index - 1].isspace()):
                    comment_start = index
                    break
            # strip full line comments
            for prefix in self._comment_prefixes:
                if line.strip().startswith(prefix):
                    comment_start = 0
                    break
            value = line[:comment_start].strip()
            if not value:
                if self._empty_lines_in_values:
                    # add empty line to the value, but only if there was no
                    # comment on the line
                    if (comment_start is None and
                            cursect is not None and
                            optname and
                            cursect[optname] is not None):
                        cursect[optname].append('')  # newlines added at join
                else:
                    # empty line marks end of value
                    indent_level = sys.maxsize
                continue
            # continuation line?
            first_nonspace = self.NONSPACECRE.search(line)
            cur_indent_level = first_nonspace.start() if first_nonspace else 0
            if (cursect is not None and optname and
                    cur_indent_level > indent_level):
                cursect[optname].append(value)
            # a section header or option header?
            else:
                indent_level = cur_indent_level
                # is it a section header?
                mo = self.SECTCRE.match(value)
                if mo:
                    sectname = mo.group('header')
                    if sectname in self._sections:
                        if self._strict and sectname in elements_added:
                            raise DuplicateSectionError(sectname, fpname,
                                                        lineno)
                        cursect = self._sections[sectname]
                        elements_added.add(sectname)
                    elif sectname == self.default_section:
                        cursect = self._defaults
                    else:
                        cursect = self._dict()
                        self._sections[sectname] = cursect
                        self._proxies[sectname] = SectionProxy(self, sectname)
                        elements_added.add(sectname)
                    # So sections can't start with a continuation line
                    optname = None
                # no section header in the file?
                elif cursect is None:
                    raise MissingSectionHeaderError(fpname, lineno, line)
                # an option line?
                else:
                    mo = self._optcre.match(value)
                    if mo:
                        optname, vi, optval = mo.group('option', 'vi', 'value')
                        if not optname:
                            e = self._handle_error(e, fpname, lineno, line)
                        optname = self.optionxform(optname.rstrip())
                        if (self._strict and
                                (sectname, optname) in elements_added):
                            raise DuplicateOptionError(sectname, optname, fpname, lineno)
                        elements_added.add((sectname, optname))
                        # This check is fine because the OPTCRE cannot
                        # match if it would set optval to None
                        if optval is not None:
                            optval = optval.strip()
                            # Check if this optname already exists
                            if (optname in cursect) and (cursect[optname] is not None):
                                # If it does, convert it to a tuple if it isn't already one
                                if not isinstance(cursect[optname], tuple):
                                    cursect[optname] = tuple(cursect[optname])
                                cursect[optname] = cursect[optname] + tuple([optval])
                            else:
                                cursect[optname] = [optval]
                        else:
                            # valueless option handling
                            cursect[optname] = None
                    else:
                        # a non-fatal parsing error occurred. set up the
                        # exception but keep going. the exception will be
                        # raised at the end of the file and will contain a
                        # list of all bogus lines
                        e = self._handle_error(e, fpname, lineno, line)
        # if any parsing errors occurred, raise an exception
        if e:
            raise e
        self._join_multiline_values()


class ControlType:
    KEYBOARD: str = "kb"
    JOYSTICK: str = "joystick"


class Control:
    def __init__(self, action: str, key: Optional[str], axis_sign: Optional[str]):
        self.action = action
        self.key = key
        self.axis_sign = axis_sign

    def __str__(self):
        return f"Action: {self.action: <{30}} | Key: {self.key}"


class ControlRecorder(ABC):
    @abstractmethod
    def get_controls(self):
        pass


class KeyboardRecorder(ControlRecorder):
    def get_controls(self):
        pass


class ControlLoader:
    def __init__(self):
        self.controls = None

    def load(self, path):
        parser = ConfigParserMultiOpt()
        parser.read(path)

        self.controls = []

        for k in parser['Legacy ControlPreset_X']['gamepadbindings']:
            attrs = k.strip()[1:-1].replace("\t","").split(",")
            for i, attr in enumerate(attrs):
                attrs[i] = attr.strip().replace("\"","").split("=")[1]
            self.controls.append(Control(
                action=attrs[0],
                key=attrs[1] if len(attrs) > 1 else None,
                axis_sign=attrs[2] if len(attrs) > 2 else None
            ))

            print(self.controls[-1])




class JoystickRecorder(ControlRecorder):
    def __init__(self, path_to_config: str, player_deadzone: float = DEFAULT_DEADZONE):
        pygame.joystick.init()
        pygame.init()

        self.path = path_to_config
        self.control = None
        self.last_jump = None
        self.player_deadzone = player_deadzone

        self._load_controls()

    def _load_controls(self):
        loader = ControlLoader()
        loader.load(self.path)

    def start(self):
        self.control = pygame.joystick.Joystick(0)
        self.control.init()
        self.last_jump = self.control.get_button(JUMP)

    def _check_for_joystick(self):
        try:
            self.control = pygame.joystick.Joystick(0)
            return True
        except Exception:
            return False

    def get_controls(self):
        pass
        pygame.event.pump()

        try:
            self.control.get_button(0)
        except Exception:
            if not self._check_for_joystick():
                return [0] * 8

        jump = self.control.get_button(JUMP)
        boost = self.control.get_button(BOOST)
        handbrake = self.control.get_button(HANDBRAKE)

        throttle = self.control.get_axis(THROTTLE)
        throttle = max(0.0, throttle)

        reverse_throttle = self.control.get_axis(REVERSE)
        reverse_throttle = max(0.0, reverse_throttle)

        throttle = throttle - reverse_throttle

        steer = self.control.get_axis(STEER_AXIS)
        if abs(steer) < self.player_deadzone:
            steer = 0.

        pitch = self.control.get_axis(YAW_AXIS)
        if abs(pitch) < self.player_deadzone:
            pitch = 0.

        yaw = steer

        roll = - self.control.get_button(AIR_ROLL_LEFT) + self.control.get_button(AIR_ROLL_RIGHT)
        roll_button = self.control.get_button(ROLL_BUTTON)
        if roll_button or (jump and not self.last_jump):
            roll = steer
            yaw = 0

        self.last_jump = jump

        return [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
