import math
import random
from typing import NamedTuple, Optional

import numpy as np
from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.math import rand_vec3
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters.wrappers.state_wrapper import StateWrapper
from rlgym_tools.extra_state_setters.wall_state import DEG_TO_RAD
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi

CAR_DIM = np.array((150, 150, 150))


def compare_arrays_inferior(arr1: np.array, arr2: np.array) -> bool:
    if arr1.shape != arr2.shape:
        print(f"Can't compare two array of different shape : {arr1.shape} and {arr2.shape}")

    return all(arr1[0:2] < arr2[0:2])


def _check_positions(state: StateWrapper, stateName) -> bool:
    # Checking all others for each car
    for car in state.cars:

        current_car = car

        for other in state.cars:
            if other == current_car:
                continue

            if compare_arrays_inferior(abs(current_car.position - other.position), CAR_DIM):
                # print(f"Car too close, resetting. State: {stateName}, Error value: {abs(current_car.position - other.position)}")
                return False

        if compare_arrays_inferior(abs(current_car.position - state.ball.position), CAR_DIM):
            return False

    return True


class RandomState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)


class ShotState(StateSetter):

    def __init__(self):
        self.alreadyBlueSet = False
        self.bluePos = None
        super().__init__()

    def createPositions(self, n_players):
        Positions = []

        for i in range(n_players - 1):
            valid_pos = False
            while (valid_pos == False):

                pos = (random.uniform(-4096, 4096), random.uniform(0, 3000), random.uniform(17, CEILING_Z - 100))

                if len(Positions) != 0:
                    for j in range((len(Positions))):
                        # print(f"j: {j}")
                        diff = np.subtract(abs(np.array(pos)), abs(np.array(Positions[j])))
                        if 200 > max(diff):
                            valid_pos = True
                            Positions.append(pos)
                            break
                else:
                    Positions.append(pos)

        print(f"len = {len(Positions)} | Positions = {Positions}")

    def reset(self, state_wrapper: StateWrapper):

        shooting_player = random.choice(state_wrapper.blue_cars)

        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM and not self.alreadyBlueSet:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, 3000),
                    17
                )

                self.alreadyBlueSet = True
                self.bluePos = car.position

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) + 1000, car.position.item(1) + 100),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            elif car.team_num == BLUE_TEAM and self.alreadyBlueSet:

                carX = self.bluePos.item(0) + 200 + random.uniform(-4096, 4096)
                carY = self.bluePos.item(1) + 200 + random.uniform(-9000, 3000)
                carZ = random.uniform(17, CEILING_Z - 100)

                if carX <= 0:
                    carX = max(-SIDE_WALL_X + 100, carX)
                else:
                    carX = min(SIDE_WALL_X - 100, carX)

                if carY <= 0:
                    carY = max(-BACK_WALL_Y + 600, carY)
                else:
                    carY = min(BACK_WALL_Y - 600, carY)

                car.set_pos(
                    carX,
                    carY,
                    carZ
                )

                self.alreadyBlueSet = False
                self.bluePos = None

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                self.alreadyBlueSet = True
                self.bluePos = car.position

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)
        '''for car in state_wrapper.cars:
            print(f"Team: {car.team_num} | Car id: {car.id} | Spawn position: {car.position}")'''


class JumpShotState(StateSetter):

    def __init__(self):
        self.alreadyBlueSet = False
        self.bluePos = None
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM and not self.alreadyBlueSet:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, 2500),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=0,
                    yaw=90,
                    roll=0
                )

                ang_vel = (0, 0, 0)  # rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) + 1500, car.position.item(1) + 500),
                    z=CEILING_Z / 2
                )

                ball_speed = np.random.uniform(100, BALL_MAX_SPEED / 2)
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 2))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = (0, 0, 0)
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class SaveState(StateSetter):

    def __init__(self):
        self.alreadyOrangeSet = False
        self.OrangePos = None
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == ORANGE_TEAM and not self.alreadyOrangeSet:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, -3000),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                self.alreadyOrangeSet = True
                self.OrangePos = car.position

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) - 1000, car.position.item(1) - 100),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            elif car.team_num == ORANGE_TEAM and self.alreadyOrangeSet:

                carX = self.OrangePos.item(0) + 200 + random.uniform(-4096, 4096)
                carY = self.OrangePos.item(1) + 200 + random.uniform(-9000, 3000)
                carZ = random.uniform(17, CEILING_Z - 100)

                if carX <= 0:
                    carX = max(-SIDE_WALL_X + 100, carX)
                else:
                    carX = min(SIDE_WALL_X - 100, carX)

                if carY <= 0:
                    carY = max(-BACK_WALL_Y + 600, carY)
                else:
                    carY = min(BACK_WALL_Y - 600, carY)

                car.set_pos(
                    carX,
                    carY,
                    carZ
                )

                self.alreadyOrangeSet = False
                self.OrangePos = None

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)
            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    -random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class AirDribble2Touch(StateSetter):

    def __init__(self):
        super().__init__()
        self.GOAL_POS = (0, GOAL_LINE, 642.775 / 2)

    def reset(self, state_wrapper: StateWrapper):

        state_choise = random.choice([1, 2])
        # state_choise = 1

        if state_choise == 0:
            for car in state_wrapper.cars:
                if car.team_num == BLUE_TEAM:
                    car_x = random.choice([-4096, 4096])
                    car_y = np.random.uniform(0, 3500)
                    car_z = np.random.uniform(300, CEILING_Z - 500)

                    car.set_pos(
                        car_x,
                        car_y,
                        car_z
                    )

                    if car_x < 0:
                        car_pitch_rot = 0.5 * math.pi
                        car_yaw_rot = 0 * math.pi
                        car_roll_rot = 1 * math.pi
                    else:
                        car_pitch_rot = 0.5 * math.pi
                        car_yaw_rot = 1 * math.pi
                        car_roll_rot = 1 * math.pi

                    car.set_rot(
                        car_pitch_rot,
                        car_yaw_rot,
                        car_roll_rot
                    )

                    car_lin_z = np.random.uniform(300, CAR_MAX_SPEED)

                    car.set_lin_vel(
                        0,
                        0,
                        car_lin_z
                    )

                    state_wrapper.ball.set_pos(
                        car_x,
                        car_y + random.uniform(160, 180),
                        car_z + random.uniform(40, 80)
                    )

                    state_wrapper.ball.set_lin_vel(
                        0,
                        0,
                        car_lin_z + random.uniform(2, 10)
                    )
                    car.boost = np.random.uniform(0.4, 1)
                else:
                    wall_choise = random.choice([1, 0])
                    if wall_choise == 0:
                        car_x = 0
                        car_y = 0
                        car_z = 0
                        for car in state_wrapper.cars:
                            if car.team_num == BLUE_TEAM:
                                car_x = car.position.item(0)
                                car_y = car.position.item(1)
                                car_z = car.position.item(2)

                        car.set_pos(
                            car_x,
                            car_y + random.uniform(200, 600),
                            car_z + random.uniform(-200, 200)
                        )

                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)

                        car.set_rot(
                            pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                            yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                            roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                        )

                        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                        car.set_ang_vel(*ang_vel)
                        car.boost = np.random.uniform(0, 1)
                    elif wall_choise == 1:
                        car.set_pos(
                            random.randint(-2900, 2900),
                            random.randint(3000, 5120),
                            17
                        )

                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)

                        car.set_rot(
                            pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                            yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                            roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                        )

                        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                        car.set_ang_vel(*ang_vel)
                        car.boost = np.random.uniform(0, 1)

        elif state_choise == 1:
            # airDribble
            for car in state_wrapper.cars:
                if car.team_num == BLUE_TEAM:
                    # handles tm8
                    if car.id != 1:
                        car.set_pos(
                            random.randint(-2900, 2900),
                            random.randint(3000, 5120),
                            17
                        )

                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)

                        car.set_rot(
                            pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                            yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                            roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                        )

                        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                        car.set_ang_vel(*ang_vel)
                        car.boost = np.random.uniform(0, 1)

                    car_x = np.random.uniform(-3500, 3500)
                    car_y = np.random.uniform(-1000, 3500)
                    car_z = np.random.uniform(1000, CEILING_Z - 500)

                    car.set_pos(
                        car_x,
                        car_y,
                        car_z
                    )

                    difference = 0
                    if car_x != 0:
                        difference = (self.GOAL_POS - car.position) / np.linalg.norm(self.GOAL_POS - car.position)

                    car_pitch_rot = random.uniform(0, 0.5) * math.pi
                    car_yaw_rot = 0.5 * math.pi
                    car_roll_rot = 0 * math.pi

                    car.set_rot(
                        car_pitch_rot,
                        car_yaw_rot,
                        car_roll_rot
                    )

                    car_speed = np.random.uniform(700, CAR_MAX_SPEED)
                    car_lin_x = car_speed * difference.item(0)
                    car_lin_y = car_speed * difference.item(1)

                    car.set_lin_vel(
                        car_lin_x,
                        car_lin_y,
                        0
                    )

                    car.set_ang_vel(0, 0, 0)

                    state_wrapper.ball.set_pos(
                        car_x,
                        car_y + random.uniform(160, 180),
                        car_z + random.uniform(40, 80)
                    )

                    ball_lin_x = car_lin_x
                    ball_lin_y = car_lin_y

                    state_wrapper.ball.set_lin_vel(
                        ball_lin_x,
                        ball_lin_y,
                        0 + random.uniform(2, 10)
                    )
                    car.boost = np.random.uniform(0.4, 1)
                else:
                    car.set_pos(
                        random.randint(-2900, 2900),
                        random.randint(3000, 5120),
                        17
                    )

                    vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                    car.set_lin_vel(*vel)

                    car.set_rot(
                        pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                        yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                        roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                    )

                    ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                    car.set_ang_vel(*ang_vel)
                    car.boost = np.random.uniform(0, 1)

        elif state_choise == 2:
            # airDribble supposed arc movement
            for car in state_wrapper.cars:
                if car.team_num == BLUE_TEAM:
                    car_x = np.random.uniform(-3500, 3500)
                    car_y = np.random.uniform(-1000, 3500)
                    car_z = np.random.uniform(300, CEILING_Z - 500)

                    car.set_pos(
                        car_x,
                        car_y,
                        car_z
                    )

                    difference = 0
                    if car_x != 0:
                        difference = (self.GOAL_POS - car.position) / np.linalg.norm(self.GOAL_POS - car.position)

                    car_pitch_rot = random.uniform(0, 0.5) * math.pi
                    car_yaw_rot = 0.5 * math.pi
                    car_roll_rot = 0 * math.pi

                    car.set_rot(
                        car_pitch_rot,
                        car_yaw_rot,
                        car_roll_rot
                    )

                    car_speed = np.random.uniform(200, CAR_MAX_SPEED - 700)
                    car_lin_x = car_speed * difference.item(0)
                    car_lin_y = car_speed * difference.item(1)

                    car.set_lin_vel(
                        car_lin_x,
                        car_lin_y,
                        0
                    )

                    car.set_ang_vel(0, 0, 0)

                    state_wrapper.ball.set_pos(
                        car_x,
                        car_y + random.uniform(160, 180),
                        car_z + random.uniform(40, 150)
                    )

                    ball_lin_x = car_lin_x
                    ball_lin_y = car_lin_y

                    state_wrapper.ball.set_lin_vel(
                        ball_lin_x,
                        ball_lin_y,
                        0 + random.uniform(2, 10)
                    )
                    car.boost = np.random.uniform(0.4, 1)

                else:
                    car.set_pos(
                        random.randint(-2900, 2900),
                        random.randint(3000, 5120),
                        17
                    )

                    vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                    car.set_lin_vel(*vel)

                    car.set_rot(
                        pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                        yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                        roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                    )

                    ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                    car.set_ang_vel(*ang_vel)
                    car.boost = np.random.uniform(0, 1)


class DefaultState(StateSetter):
    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        rand_default_or_diff = random.randint(0, 3)
        # rand_default_or_diff = 0
        # DEFAULT KICKOFFS POSITIONS
        if rand_default_or_diff == 0:

            # possible kickoff indices are shuffled
            spawn_inds = [0, 1, 2, 3, 4]
            random.shuffle(spawn_inds)

            blue_count = 0
            orange_count = 0
            for car in state_wrapper.cars:
                pos = [0, 0, 0]
                yaw = 0
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                    yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                    blue_count += 1
                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                    yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                    orange_count += 1
                # set car state values
                car.set_pos(*pos)
                car.set_rot(yaw=yaw)
                car.boost = 0.33
        else:
            """SPAWN POS RANGE
            Y = CENTER BACK KICKOFF [-4608, 0], [4608, 0]; BACK LEFT/RIGHT KICKOFFS [-3000, 0], [3000,0]"""

            # print("Advanced Kickoffs")
            # SPAWN POSITION
            spawn_inds = [0, 1, 2, 3, 4]
            spawn_pos = np.random.choice(spawn_inds)

            same_pos = 0
            # print(f"Same position = {same_pos}")

            # RIGHT CORNER
            if spawn_pos == 0:
                # print("Right Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / -2048
                    posX = np.random.uniform(-2048, -200)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 200)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / -2048
                        blue_posX = np.random.uniform(-2048, -200)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / 2048
                        orange_posX = np.random.uniform(2048, 200)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            # LEFT CORNER
            elif spawn_pos == 1:
                # print("Left Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / 2048
                    posX = np.random.uniform(2048, 200)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / 2048
                        blue_posX = np.random.uniform(2048, 200)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / -2048
                        orange_posX = np.random.uniform(-2048, -200)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 2:
                # print("Back Right")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / -256
                    posX = np.random.uniform(-256, -50)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (0, posY, posZ)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / -256
                        blue_posX = np.random.uniform(-256, -50)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / 256
                        orange_posX = np.random.uniform(256, 0)
                        orange_posY = slope * orange_posX
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 3:
                # print("Back Left")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / 256
                    posX = np.random.uniform(256, 50)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (posX, 0, 17)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / 256
                        blue_posX = np.random.uniform(256, 50)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / -256
                        orange_posX = np.random.uniform(-256, -50)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 4:
                # print("Far Back Center")
                # SAME POS
                if same_pos == 1:
                    posX = 0
                    posY = np.random.uniform(-4608, -200)
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (posX, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        blue_posX = 0
                        blue_posY = np.random.uniform(-4608, -200)
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        orange_posX = 0
                        orange_posY = np.random.uniform(4608, 0)
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)


class DribbleSetup(StateSetter):

    def __init__(self):
        self.alreadyBlueSet = False
        self.bluePos = None
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            # if car.team_num == BLUE_TEAM and not self.alreadyBlueSet:
            if car.team_num == BLUE_TEAM and car.id == 1:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(-4000, 4000),
                    17
                )

                self.alreadyBlueSet = True
                self.bluePos = car.position

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=0,
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=0,
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=car.position.item(0),
                    y=car.position.item(1),
                    z=car.position.item(2) + BALL_RADIUS * 1.5,
                )
                state_wrapper.ball.set_lin_vel(*vel)
                state_wrapper.ball.set_ang_vel(*ang_vel)

            # elif car.team_num == BLUE_TEAM and self.alreadyBlueSet:
            elif car.team_num == BLUE_TEAM and car.team_num == 2:
                carX = self.bluePos.item(0) + 200 + random.uniform(-4096, 4096)
                carY = self.bluePos.item(1) + 200 + random.uniform(-9000, 3000)
                carZ = random.uniform(17, CEILING_Z - 100)

                if carX <= 0:
                    carX = max(-SIDE_WALL_X + 100, carX)
                else:
                    carX = min(SIDE_WALL_X - 100, carX)

                if carY <= 0:
                    carY = max(-BACK_WALL_Y + 600, carY)
                else:
                    carY = min(BACK_WALL_Y - 600, carY)

                car.set_pos(
                    carX,
                    carY,
                    carZ
                )

                self.alreadyBlueSet = False
                self.bluePos = None

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                self.alreadyBlueSet = True
                self.bluePos = car.position

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class AirDribbleSetup(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        axis_inverter = 1 if random.randrange(2) == 1 else -1
        team_side = 0 if random.randrange(2) == 1 else 1
        team_inverter = 1 if team_side == 0 else -1

        # if only 1 play, team is always 0

        ball_x_pos = 3000 * axis_inverter
        ball_y_pos = random.randrange(7600) - 3800
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + (random.randrange(1000) - 500)) * axis_inverter
        ball_y_vel = random.randrange(1000) * team_inverter
        ball_z_vel = 0
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        chosen_car = [car for car in state_wrapper.cars if car.team_num == team_side][0]
        # if randomly pick, chosen_car is from orange instead

        car_x_pos = 2500 * axis_inverter
        car_y_pos = ball_y_pos
        car_z_pos = 27

        yaw = 0 if axis_inverter == 1 else 180
        car_pitch_rot = 0 * DEG_TO_RAD
        car_yaw_rot = (yaw + (random.randrange(40) - 20)) * DEG_TO_RAD
        car_roll_rot = 0 * DEG_TO_RAD

        chosen_car.set_pos(car_x_pos, car_y_pos, car_z_pos)
        chosen_car.set_rot(car_pitch_rot, car_yaw_rot, car_roll_rot)
        chosen_car.boost = 1

        for car in state_wrapper.cars:
            if car is chosen_car:
                continue
            else:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    random.randint(20, CEILING_Z - 300),
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

            # set all other cars randomly in the field


class SideHighRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = random.randrange(300)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]

        # blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 600 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 1

        # orange car setup
        wall_car_orange = None
        if len(state_wrapper.cars) > 1:
            wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]
            # orange car setup
            orange_pitch_rot = 0 * DEG_TO_RAD
            orange_yaw_rot = -90 * DEG_TO_RAD
            orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
            wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

            orange_x = 4096 * side_inverter
            orange_y = 2500 + (random.randrange(500) - 250)
            orange_z = 400 + (random.randrange(400) - 200)
            wall_car_orange.set_pos(orange_x, orange_y, orange_z)
            wall_car_orange.boost = 1

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)


class ShortGoalRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        if len(state_wrapper.cars) > 1:
            defense_team = random.randrange(2)
        else:
            defense_team = 0
        sidepick = random.randrange(2)

        defense_inverter = 1
        if defense_team == 0:
            # change side
            defense_inverter = -1

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        x_random = random.randrange(446)
        ball_x_pos = (-2850 + x_random) * side_inverter
        ball_y_pos = (5120 - BALL_RADIUS) * defense_inverter
        ball_z_pos = 1400 + random.randrange(400) - 200
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (1000 + random.randrange(400) - 200) * side_inverter
        ball_y_vel = 0
        ball_z_vel = 550
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car = [car for car in state_wrapper.cars if car.team_num == defense_team][0]

        wall_car_x = (2000 - random.randrange(500)) * side_inverter
        wall_car_y = 5120 * defense_inverter
        wall_car_z = 1000 + (random.randrange(500) - 500)
        wall_car.set_pos(wall_car_x, wall_car_y, wall_car_z)

        wall_pitch_rot = (0 if side_inverter == -1 else 180) * DEG_TO_RAD
        wall_yaw_rot = 0 * DEG_TO_RAD
        wall_roll_rot = -90 * defense_inverter * DEG_TO_RAD
        wall_car.set_rot(wall_pitch_rot, wall_yaw_rot, wall_roll_rot)
        wall_car.boost = 0.25

        if len(state_wrapper.cars) > 1:
            challenge_car = [car for car in state_wrapper.cars if car.team_num != defense_team][0]
            challenge_car.set_pos(0, 1000 * defense_inverter, 0)

            challenge_pitch_rot = 0 * DEG_TO_RAD
            challenge_yaw_rot = 90 * defense_inverter * DEG_TO_RAD
            challenge_roll_rot = 0 * DEG_TO_RAD
            challenge_car.set_rot(challenge_pitch_rot, challenge_yaw_rot, challenge_roll_rot)
            challenge_car.boost = 1

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car or car is challenge_car:
                continue

            car.set_pos(random.randrange(2944) - 1472, (-4500 + random.randrange(500) - 250) * defense_inverter, 0)
            car.set_rot(0, (random.randrange(360) - 180) * DEG_TO_RAD, 0)


class StandingBallState(StateSetter):

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(x=random.Random().randint(-SIDE_WALL_X + 500, SIDE_WALL_X - 500),
                                   y=0,
                                   z=random.Random().randint(1000, 1300))

        state_wrapper.ball.set_lin_vel(x=random.uniform(-1000, 1000),
                                       y=random.uniform(-1000, 1000),
                                       z=random.uniform(400, 800)
                                       )

        blue_team = []
        orange_team = []

        for player in state_wrapper.cars:

            x = 0
            y = 0
            yaw = 0
            if player.team_num == ORANGE_TEAM:

                y = random.uniform(1000, 2000)
                yaw = -0.5 * np.pi

                if len(orange_team) >= 1:
                    choice = random.choice(orange_team)

                    if choice.position.item(0) > 3000:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) - 200)

                    elif choice.position.item(0) < -3000:
                        x = random.Random().randint(choice.position.item(0) + 200,
                                                    choice.position.item(0) + 500)
                    else:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) + 500)

                else:
                    x = random.Random().randint(state_wrapper.ball.position.item(0) - 300,
                                                state_wrapper.ball.position.item(0) + 300)

                orange_team.append(player)
            else:
                y = random.uniform(-2000, -1000)
                yaw = 0.5 * np.pi

                if len(blue_team) >= 1:
                    choice = random.choice(blue_team)

                    if choice.position.item(0) > 3000:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) - 200)

                    elif choice.position.item(0) < -3000:
                        x = random.Random().randint(choice.position.item(0) + 200,
                                                    choice.position.item(0) + 500)
                    else:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) + 500)

                else:
                    x = random.Random().randint(
                        state_wrapper.ball.position.item(0) - 300,
                        state_wrapper.ball.position.item(0) + 300)

            player.set_pos(
                x=x,
                y=y,
                z=30)
            player.set_rot(yaw=yaw)
            player.boost = 1


class AerialBallState(StateSetter):
    def __init__(self):
        super().__init__()
        self.observators = []

    def reset(self, state_wrapper: StateWrapper):

        if len(self.observators) != 0:
            distances = []
            for player in state_wrapper.cars:
                dist = math.sqrt((state_wrapper.ball.position.item(0) - player.position.item(0)) ** 2 +
                                 (state_wrapper.ball.position.item(1) - player.position.item(1)) ** 2 +
                                 (state_wrapper.ball.position.item(2) - player.position.item(2)) ** 2)
                distances.append(dist)

            distance = np.mean(distances)

            for obs in self.observators:
                obs.update(distance=float(distance))

        xthreshold = 1800
        ythreshold = 1200
        zthreshold = 500

        # Ball position
        state_wrapper.ball.set_pos(x=np.random.randint(-SIDE_WALL_X + xthreshold,
                                                       SIDE_WALL_X - xthreshold),

                                   y=np.random.randint(-BACK_WALL_Y + ythreshold,
                                                       BACK_WALL_Y - ythreshold),

                                   z=np.random.randint(zthreshold,
                                                       CEILING_Z - 2 * zthreshold)
                                   )

        # Ball speed
        state_wrapper.ball.set_lin_vel(x=np.random.randint(500, 1000),
                                       y=np.random.randint(500, 1000),
                                       z=np.random.randint(500, 1000)
                                       )

        for player in state_wrapper.cars:
            player.set_pos(x=np.random.randint(-SIDE_WALL_X + xthreshold,
                                               SIDE_WALL_X - xthreshold),
                           y=np.random.randint(-BACK_WALL_Y + ythreshold,
                                               BACK_WALL_Y - ythreshold),
                           z=30)

            player.boost = 1


class StateResetResult(NamedTuple):
    state: StateWrapper
    error: Optional[Exception]


class WeightedVerifiedStateSetter(WeightedSampleSetter):

    def reset(self, state_wrapper: StateWrapper):
        try:
            super().reset(state_wrapper)
            return StateResetResult(state_wrapper, None)
        except Exception as e:
            return StateResetResult(state_wrapper, e)
