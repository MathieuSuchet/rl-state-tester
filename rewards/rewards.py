import math

import numpy as np
from numpy import ndarray
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward


class BumpReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 0 = Blue; 1 = Orange

        if player.on_ground:
            for agent in state.players:
                if agent.team_num == player.team_num:
                    relPos = player.car_data.position - agent.car_data.position
                    distToAgent = np.linalg.norm(relPos)

                    if distToAgent < 250 and distToAgent > 1:
                        # print(f"Bumped.  Distance to Agent: {np.round(distToAgent,5)}")
                        return -8
                    return 0
                elif agent.team_num != player.team_num:
                    relPos = player.car_data.position - agent.car_data.position
                    distToAgent = np.linalg.norm(relPos)

                    if distToAgent < 170 and distToAgent > 1:
                        # print(f"Opponent Bumped.  Distance to Agent: {np.round(distToAgent,5)}")
                        return 0.1
                return 0
        return 0


class TestReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.car_id == 1:
            print(player.boost_amount)
        return 0


class DistToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        distance = 1 / math.sqrt((state.ball.position.item(0) - player.car_data.position.item(0)) ** 2 +
                                 (state.ball.position.item(1) - player.car_data.position.item(1)) ** 2 +
                                 (state.ball.position.item(2) - player.car_data.position.item(2)) ** 2)

        return distance * float(100)


class PlayerVelocityReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        reward = 0

        player_vel = np.linalg.norm(player.car_data.linear_velocity)
        reward = player_vel / CAR_MAX_SPEED

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        return 0


class ClosestDistToBallReward(RewardFunction):
    # ENCOURAGE CONTROLLING THE BALL AND KEEPING THE BALL ON THE TOP OF THE CAR
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        touchHeight = TouchHeightReward.get_reward(self, player, state, previous_action) * 10

        agent = None
        for p in state.players:
            if agent == None:
                agent = p
            else:
                if np.linalg.norm(p.car_data.position - state.ball.position) < np.linalg.norm(
                        agent.car_data.position - state.ball.position):
                    agent = p

        if agent is player:
            distance = 1 / math.sqrt((state.ball.position.item(0) - player.car_data.position.item(0)) ** 2 +
                                     (state.ball.position.item(1) - player.car_data.position.item(1)) ** 2 +
                                     (state.ball.position.item(2) - player.car_data.position.item(2)) ** 2)
            reward = distance * float(100)

        reward += touchHeight
        return reward


class DistancePlayerToBall(RewardFunction):
    def __init__(self):
        self.data = []

    def reset(self, initial_state: GameState):
        if len(self.data) != 0:
            self.data.clear()
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        self.data.append(dist)

        return -(dist ** 2) / 6_000_000 + 0.5


class FlipVelReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        playerVel = np.linalg.norm(player.car_data.linear_velocity)

        if playerVel > SUPERSONIC_THRESHOLD - 400:
            reward = 0
            if not player.has_flip:
                reward += 1
            if playerVel > SUPERSONIC_THRESHOLD:
                reward += 0.1
            reward += player.boost_amount

            return reward + 0.1
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class GoalScoreSpeed(RewardFunction):
    def __init__(self) -> None:
        self.orangeScore = 0
        self.blueScore = 0

    def reset(self, initial_state: GameState):
        self.orangeScore = initial_state.orange_score
        self.blueScore = initial_state.blue_score
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ballSpeed = np.linalg.norm(state.ball.linear_velocity)
        if player.team_num == BLUE_TEAM:
            if self.blueScore != state.blue_score:
                kmhReward = 215 * (ballSpeed / BALL_MAX_SPEED)
                reward = max(0, 0.5 * np.log2(kmhReward / 10) - 0.6)
                # print(f"Blue reward: {reward}")
                return reward  # blue scored
        if player.team_num == ORANGE_TEAM:
            if self.orangeScore != state.orange_score:
                kmhReward = 215 * (ballSpeed / BALL_MAX_SPEED)
                reward = max(0, 0.5 * np.log2(kmhReward / 10) - 0.6)
                # print(f"Orange reward: {reward}")
                return reward  # orange scored
        return 0


class EpisodeLengthReward(RewardFunction):
    def __init__(self):
        self.nb_steps_since_reset = 0

    def reset(self, initial_state: GameState):
        self.nb_steps_since_reset = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.nb_steps_since_reset += 1

        return - self.nb_steps_since_reset ** 2 / 25_000 + 1

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)


# region KickOff

# class LeftKickoffReward(RewardFunction):
class KickoffReward(RewardFunction):

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: ndarray
    ) -> float:
        reward = 0
        if state.ball.position.item(0) == 0:
            reward += VelocityPlayerToBallReward.get_reward(VelocityPlayerToBallReward(), player, state,
                                                            previous_action) * 10
        if state.ball.position[1] > 500:
            # print("KICKOFF WON BY BLUE")
            if player.team_num == BLUE_TEAM:
                reward += 10
            else:
                reward -= 10

            return reward
        if state.ball.position[1] < -500:
            # print("KICKOFF WON BY ORANGE")
            if player.team_num == ORANGE_TEAM:
                reward += 10
            else:
                reward -= 10
            return reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        return 0


class KickoffReward_MMR(RewardFunction):
    def __init__(self):
        self.KickoffReward = KickoffReward()
        self.blueTeam = []
        self.orangeTeam = []
        self.last_pos_saved = False
        self.is_Kickoff = False
        self.is_Kickoff_Done = False
        self.closest_blue_agent = None
        self.closest_orange_agent = None

    def reset(self, initial_state: GameState):
        self.last_pos_saved = False
        self.is_Kickoff = False
        self.is_Kickoff_Done = False
        self.closest_blue_agent = None
        self.closest_orange_agent = None
        return 0

    def get_Players(self, players):
        for p in players:
            if p.team_num == BLUE_TEAM:
                self.blueTeam.append(p)
            else:
                self.orangeTeam.append(p)

    def get_Closest_Blue(self, ballPos):
        agent = None
        for p in self.blueTeam:
            if agent == None:
                agent = p
            else:
                if np.linalg.norm(p.car_data.position - ballPos) < np.linalg.norm(agent.car_data.position - ballPos):
                    agent = p
        return agent

    def get_Closest_Orange(self, ballPos):
        agent = None
        for p in self.orangeTeam:
            if agent == None:
                agent = p
            else:
                if np.linalg.norm(p.car_data.position - ballPos) < np.linalg.norm(agent.car_data.position - ballPos):
                    agent = p
        return agent

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:

        if len(self.blueTeam) == 0:
            self.get_Players(state.players)

        if self.closest_blue_agent is None:
            self.closest_blue_agent = self.get_Closest_Blue(state.ball.position)

        if self.closest_orange_agent is None:
            self.closest_orange_agent = self.get_Closest_Orange(state.ball.position)

        # Is KICKOFF
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and not self.last_pos_saved:
            self.last_pos_saved = True

        if self.last_pos_saved and not self.is_Kickoff and state.ball.position[0] == 0 and state.ball.position[1] == 0:
            self.is_Kickoff = True

        if self.is_Kickoff and not self.is_Kickoff_Done:
            if state.ball.position[1] < -500 or state.ball.position[1] > 500:
                self.is_Kickoff_Done = True
            if player.car_id == self.closest_orange_agent.car_id or player.car_id == self.closest_blue_agent.car_id:
                return self.KickoffReward.get_reward(player, state, previous_action)
        return 0


# endregion

# region Boost
class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.on_ground:

            if state.ball.position[0] == 0 and state.ball.position[1] == 0:
                return 0

            reward = 0
            try:
                reward = (0.1 * math.log2(player.boost_amount - 0.06) + 0.4) / 10
            except ValueError:
                reward = 0
            return reward
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        return 0


class BoostPickupReward(RewardFunction):
    def __init__(self):
        self.prev_boost = 0

    def reset(self, initial_state: GameState):
        self.prev_boost = 0
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if self.prev_boost < player.boost_amount:
            if player.boost_amount - self.prev_boost < 15:
                reward = 0.8
            else:
                reward = 1
            self.prev_boost = player.boost_amount

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


# endregion

class DribbleReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        if player.on_ground:
            if state.ball.position.item(0) > -4000 or state.ball.position.item(0) < 4000 or (
                    state.ball.position.item(1) < 5020 and 900 > state.ball.position.item(0) < -900) or (
                    state.ball.position.item(1) > -5020 and 900 < state.ball.position.item(0) < -900):
                if 180 > state.ball.position[2] > 130 and player.ball_touched:
                    return 10
        return 0


class AerialReward(RewardFunction):

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        reward = 0
        if state.ball.position[2] > 230:  # 230
            if player.ball_touched:
                # aerial touch
                reward = 1

                # for power flicks while dribbling
                if 400 > state.ball.position[2] < 220:
                    ballSpeedReward = np.linalg.norm(state.ball.linear_velocity) / BALL_MAX_SPEED
                    reward += ballSpeedReward
                    # print(f"heightReward: {heightReward}")

                # if no jump we reward ball power
                if not player.has_jump:
                    ballSpeedReward = (np.linalg.norm(state.ball.linear_velocity) / BALL_MAX_SPEED) / 10
                    reward += ballSpeedReward
                    # print(f"BallSpeedReward: {ballSpeedReward}")

                reward += state.ball.position[2] / CEILING_Z
                return reward
            elif player.car_data.position.item(2) > 300:
                dist = DistToBallReward.get_reward(DistToBallReward(), player, state, previous_action)
                if dist < 0.6:
                    reward += state.ball.position[2] / CEILING_Z
                    # distToBall
                    reward += dist * 30

            return reward
        return 0


class WallReward(RewardFunction):

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # reward for distance to ball + wall encouragment + after wall near the ball reward
        reward = 0

        # starting encouragement
        if state.ball.position.item(0) < -3900 or state.ball.position.item(0) > 3900 or (
                state.ball.position.item(1) > 5000 and 900 < state.ball.position.item(0) > -900) or (
                state.ball.position.item(1) < -5000 and 900 < state.ball.position.item(0) > -900):

            distToBallReward = DistToBallReward.get_reward(DistToBallReward(), player, state, previous_action) * 0.04
            velToBallReward = VelocityPlayerToBallReward.get_reward(VelocityPlayerToBallReward(), player, state,
                                                                    previous_action) * 30

            reward += (distToBallReward * 2 + velToBallReward) / 10

            if player.ball_touched:
                reward += 10 + (player.car_data.position.item(2) / CEILING_Z)

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class FlipReward(RewardFunction):
    def __init__(self) -> None:

        self.Player1 = {
            "lastSpeed": 0,
            "isInAir": False,
        }
        self.Player2 = {
            "lastSpeed": 0,
            "isInAir": False,
        }
        self.Player4 = {
            "lastSpeed": 0,
            "isInAir": False,
        }
        self.Player5 = {
            "lastSpeed": 0,
            "isInAir": False,
        }
        self.maxPlayerChangeSpeed = 1300

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:

        if player.car_id == 1:
            reward = 0

            curSpeed = np.linalg.norm(player.car_data.linear_velocity)

            # if player is on ground (has flipped) and it was in the air we gave it reward if it gained speed and set inair to false
            if player.on_ground and self.Player1["isInAir"]:
                if self.Player1["lastSpeed"] < curSpeed:
                    reward = abs(curSpeed - self.Player1["lastSpeed"]) / self.maxPlayerChangeSpeed
                    # reward -= (player.boost_amount - self.lastBoost_1) / 100
                    # print(f"Player_ID: {player.car_id} | Speed Gain: {curSpeed - self.lastSpeed_1} | Reward: {reward}")
                self.Player1.update({"isInAir": False})

            # if player is not on ground but we still dk it's in the air we save last speed and set inair
            if not player.on_ground and not self.Player1["isInAir"]:
                distToBallReward = DistToBallReward.get_reward(DistToBallReward(), player, state, previous_action)
                if distToBallReward < 0.1:
                    self.Player1.update({"lastSpeed": curSpeed})
                    self.Player1.update({"isInAir": True})

            return reward
        if player.car_id == 2:
            reward = 0
            curSpeed = np.linalg.norm(player.car_data.linear_velocity)

            # if player is on ground (has flipped) and it was in the air we gave it reward if it gained speed and set inair to false
            if player.on_ground and self.Player2["isInAir"]:
                if self.Player2["lastSpeed"] < curSpeed:
                    reward = abs(curSpeed - self.Player2["lastSpeed"]) / self.maxPlayerChangeSpeed
                    # print(f"Player_ID: {player.car_id} | Speed Gain: {curSpeed - self.lastSpeed_2} | Reward: {reward}")
                self.Player2.update({"isInAir": False})

            # if player is not on ground but we still dk it's in the air we save last speed and set inair
            if not player.on_ground and not self.Player2["isInAir"]:
                distToBallReward = DistToBallReward.get_reward(DistToBallReward(), player, state, previous_action)
                if distToBallReward < 0.1:
                    self.Player2.update({"lastSpeed": curSpeed})
                    self.Player2.update({"isInAir": True})

            return reward

        if player.car_id == 4:
            reward = 0
            curSpeed = np.linalg.norm(player.car_data.linear_velocity)

            # if player is on ground (has flipped) and it was in the air we gave it reward if it gained speed and set inair to false
            if player.on_ground and self.Player4["isInAir"]:
                if self.Player4["lastSpeed"] < curSpeed:
                    reward = abs(curSpeed - self.Player4["lastSpeed"]) / self.maxPlayerChangeSpeed
                    # print(f"Player_ID: {player.car_id} | Speed Gain: {curSpeed - self.lastSpeed_4} | Reward: {reward}")
                self.Player4.update({"isInAir": False})

            # if player is not on ground but we still dk it's in the air we save last speed and set inair
            if not player.on_ground and not self.Player4["isInAir"]:
                distToBallReward = DistToBallReward.get_reward(DistToBallReward(), player, state, previous_action)
                if distToBallReward < 0.1:
                    self.Player4.update({"lastSpeed": curSpeed})
                    self.Player4.update({"isInAir": True})

            return reward

        if player.car_id == 5:
            reward = 0
            curSpeed = np.linalg.norm(player.car_data.linear_velocity)
            # if player is on ground (has flipped) and it was in the air we gave it reward if it gained speed and set inair to false
            if player.on_ground and self.Player5["isInAir"]:
                if self.Player5["lastSpeed"] < curSpeed:
                    reward = abs(curSpeed - self.Player5["lastSpeed"]) / self.maxPlayerChangeSpeed
                    # print(f"Player_ID: {player.car_id} | Speed Gain: {curSpeed - self.lastSpeed_5} | Reward: {reward}")
                self.Player5.update({"isInAir": False})

            # if player is not on ground but we still dk it's in the air we save last speed and set inair
            if not player.on_ground and not self.Player5["isInAir"]:
                distToBallReward = DistToBallReward.get_reward(DistToBallReward(), player, state, previous_action)
                if distToBallReward < 0.1:
                    self.Player5.update({"lastSpeed": curSpeed})
                    self.Player5.update({"isInAir": True})

            return reward

        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class WasteSpeedReward(RewardFunction):
    def __init__(self) -> None:
        self.Player1 = {
            "initialPos": 0,
            "initialSpeed": 0,
            "initialBoost": 0,
        }
        self.Player2 = {
            "initialPos": 0,
            "initialSpeed": 0,
            "initialBoost": 0,
        }
        self.Player4 = {
            "initialPos": 0,
            "initialSpeed": 0,
            "initialBoost": 0,
        }
        self.Player5 = {
            "initialPos": 0,
            "initialSpeed": 0,
            "initialBoost": 0,
        }

        self.moveOffset = 250
        self.maxPlayerChangeSpeed = 1300
        self.distToBall = DistToBallReward()
        self.distToBallReward = 0

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:

        if player.on_ground:
            if player.car_id == 1:
                self.distToBallReward = self.distToBall.get_reward(player, state, previous_action)

                # check pos passing position 0 and vel
                if self.checkPos(abs(player.car_data.position), np.linalg.norm(player.car_data.linear_velocity),
                                 player.boost_amount, player):

                    vel = np.linalg.norm(player.car_data.linear_velocity)
                    boost = player.boost_amount

                    boostUsage = 0
                    if abs(self.Player1["initialBoost"] - boost) >= 0:
                        boostUsage = self.Player1["initialBoost"] - boost
                    else:
                        boostUsage = -1

                    velGain = (vel - self.Player1["initialSpeed"])

                    # driving with no boost
                    if velGain < 300 and boost == 0 and player.on_ground:
                        '''if player.car_id == 1:
                            print(f"Driving while having NO boost. Speed Gained: {velGain:1f}")'''
                        return -0.1

                    # wasting boost and gaining no speed
                    if velGain < 300 and boostUsage > 0:
                        ''' if player.car_id == 1:
                            print(f"Using boost while gaining LOW speed. Reward is negative. Speed Gained: {velGain:1f} | Boost wasted: {boostUsage:1f}")'''
                        return -0.5
                    # high boost usage
                    if velGain > 300 and boostUsage > 0.20:
                        # print(f"Gained speed, but WASTED a lot of boost. Speed Gained: {velGain:1f} |  Boost wasted {boostUsage:1f}")
                        return -1

                    # print(f"Changed position 0 from {self.initialPos0:2f} to {pos:2f} | Changed velocity from {self.initialSpeed0:2f} to {vel:2f}. Velocity Gain is {velGain:2f} | Changed boost from {self.initialBoost0:2f} to {boost:2f}. Boost difference is {(self.initialBoost0 - boost):2f}")
                    self.resetVars()
                    return 0
            if player.car_id == 2:
                self.distToBallReward = self.distToBall.get_reward(player, state, previous_action)

                # check pos passing position 0 and vel
                if self.checkPos(abs(player.car_data.position), np.linalg.norm(player.car_data.linear_velocity),
                                 player.boost_amount, player):

                    vel = np.linalg.norm(player.car_data.linear_velocity)
                    boost = player.boost_amount

                    boostUsage = 0
                    if abs(self.Player2["initialBoost"] - boost) >= 0:
                        boostUsage = self.Player2["initialBoost"] - boost
                    else:
                        boostUsage = -1

                    velGain = (vel - self.Player2["initialSpeed"])

                    # driving with no boost
                    if velGain < 300 and boost == 0 and player.on_ground:
                        '''if player.car_id == 1:
                            print(f"Driving while having NO boost. Speed Gained: {velGain:1f}")'''
                        return -0.1

                    # wasting boost and gaining no speed
                    if velGain < 300 and boostUsage > 0:
                        ''' if player.car_id == 1:
                            print(f"Using boost while gaining LOW speed. Reward is negative. Speed Gained: {velGain:1f} | Boost wasted: {boostUsage:1f}")'''
                        return -0.5
                    # high boost usage
                    if velGain > 300 and boostUsage > 0.20:
                        # print(f"Gained speed, but WASTED a lot of boost. Speed Gained: {velGain:1f} |  Boost wasted {boostUsage:1f}")
                        return -1

                    # print(f"Changed position 0 from {self.initialPos0:2f} to {pos:2f} | Changed velocity from {self.initialSpeed0:2f} to {vel:2f}. Velocity Gain is {velGain:2f} | Changed boost from {self.initialBoost0:2f} to {boost:2f}. Boost difference is {(self.initialBoost0 - boost):2f}")
                    self.resetVars()
                    return 0
            if player.car_id == 4:
                self.distToBallReward = self.distToBall.get_reward(player, state, previous_action)

                # check pos passing position 0 and vel
                if self.checkPos(abs(player.car_data.position), np.linalg.norm(player.car_data.linear_velocity),
                                 player.boost_amount, player):

                    vel = np.linalg.norm(player.car_data.linear_velocity)
                    boost = player.boost_amount

                    boostUsage = 0
                    if abs(self.Player4["initialBoost"] - boost) >= 0:
                        boostUsage = self.Player4["initialBoost"] - boost
                    else:
                        boostUsage = -1

                    velGain = (vel - self.Player4["initialSpeed"])

                    # driving with no boost
                    if velGain < 300 and boost == 0 and player.on_ground:
                        '''if player.car_id == 1:
                            print(f"Driving while having NO boost. Speed Gained: {velGain:1f}")'''
                        return -0.1

                    # wasting boost and gaining no speed
                    if velGain < 300 and boostUsage > 0:
                        ''' if player.car_id == 1:
                            print(f"Using boost while gaining LOW speed. Reward is negative. Speed Gained: {velGain:1f} | Boost wasted: {boostUsage:1f}")'''
                        return -0.5
                    # high boost usage
                    if velGain > 300 and boostUsage > 0.20:
                        # print(f"Gained speed, but WASTED a lot of boost. Speed Gained: {velGain:1f} |  Boost wasted {boostUsage:1f}")
                        return -1

                    # print(f"Changed position 0 from {self.initialPos0:2f} to {pos:2f} | Changed velocity from {self.initialSpeed0:2f} to {vel:2f}. Velocity Gain is {velGain:2f} | Changed boost from {self.initialBoost0:2f} to {boost:2f}. Boost difference is {(self.initialBoost0 - boost):2f}")
                    self.resetVars()
                    return 0
            if player.car_id == 5:
                self.distToBallReward = self.distToBall.get_reward(player, state, previous_action)

                # check pos passing position 0 and vel
                if self.checkPos(abs(player.car_data.position), np.linalg.norm(player.car_data.linear_velocity),
                                 player.boost_amount, player):

                    vel = np.linalg.norm(player.car_data.linear_velocity)
                    boost = player.boost_amount

                    boostUsage = 0
                    if abs(self.Player5["initialBoost"] - boost) >= 0:
                        boostUsage = self.Player5["initialBoost"] - boost
                    else:
                        boostUsage = -1

                    velGain = (vel - self.Player5["initialSpeed"])

                    # driving with no boost
                    if velGain < 300 and boost == 0 and player.on_ground:
                        '''if player.car_id == 1:
                            print(f"Driving while having NO boost. Speed Gained: {velGain:1f}")'''
                        return -0.1

                    # wasting boost and gaining no speed
                    if velGain < 300 and boostUsage > 0:
                        ''' if player.car_id == 1:
                            print(f"Using boost while gaining LOW speed. Reward is negative. Speed Gained: {velGain:1f} | Boost wasted: {boostUsage:1f}")'''
                        return -0.5
                    # high boost usage
                    if velGain > 300 and boostUsage > 0.20:
                        # print(f"Gained speed, but WASTED a lot of boost. Speed Gained: {velGain:1f} |  Boost wasted {boostUsage:1f}")
                        return -1

                    # print(f"Changed position 0 from {self.initialPos0:2f} to {pos:2f} | Changed velocity from {self.initialSpeed0:2f} to {vel:2f}. Velocity Gain is {velGain:2f} | Changed boost from {self.initialBoost0:2f} to {boost:2f}. Boost difference is {(self.initialBoost0 - boost):2f}")
                    self.resetVars()
                    return 0
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        return 0

    def checkPos(self, position, vel, boost, player):
        # set initial pos and speed if they are not set and return FALSE

        if player.car_id == 1:

            if np.all(self.Player1["initialPos"]) == 0 and np.all(self.Player1["initialSpeed"]) == 0:
                self.Player1.update({"initialPos": position, "initialBoost": boost, "initialSpeed": vel})
                return False

            # if initial pos is set we look if the bot moved enough
            if np.linalg.norm(
                    abs(np.subtract(self.Player1["initialPos"], position))) > self.moveOffset and player.on_ground:
                # bot moved enough, returns TRUE to be rewarded
                return True

        if player.car_id == 2:

            if np.all(self.Player2["initialPos"]) == 0 and np.all(self.Player2["initialSpeed"]) == 0:
                self.Player2.update({"initialPos": position, "initialBoost": boost, "initialSpeed": vel})
                return False
            # if initial pos is set we look if the bot moved enough
            if np.linalg.norm(
                    abs(np.subtract(self.Player2["initialPos"], position))) > self.moveOffset and player.on_ground:
                # bot moved enough, returns TRUE to be rewarded
                return True

        if player.car_id == 4:

            if np.all(self.Player4["initialPos"]) == 0 and np.all(self.Player4["initialSpeed"]) == 0:
                self.Player4.update({"initialPos": position, "initialBoost": boost, "initialSpeed": vel})
                return False

            # if initial pos is set we look if the bot moved enough
            if np.linalg.norm(
                    abs(np.subtract(self.Player4["initialPos"], position))) > self.moveOffset and player.on_ground:
                # bot moved enough, returns TRUE to be rewarded
                return True

        if player.car_id == 5:

            if np.all(self.Player5["initialPos"]) == 0 and np.all(self.Player5["initialSpeed"]) == 0:
                self.Player5.update({"initialPos": position, "initialBoost": boost, "initialSpeed": vel})
                return False

            # if initial pos is set we look if the bot moved enough
            if np.linalg.norm(
                    abs(np.subtract(self.Player5["initialPos"], position))) > self.moveOffset and player.on_ground:
                # bot moved enough, returns TRUE to be rewarded
                return True

    def resetVars(self):
        self.Player1.update({"initialPos": 0, "initialBoost": 0, "initialSpeed": 0})
        self.Player2.update({"initialPos": 0, "initialBoost": 0, "initialSpeed": 0})
        self.Player5.update({"initialPos": 0, "initialBoost": 0, "initialSpeed": 0})
        self.Player5.update({"initialPos": 0, "initialBoost": 0, "initialSpeed": 0})


class TouchHeightReward(RewardFunction):

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.ball_touched and state.ball.position.item(2) > BALL_RADIUS:
            playerHeight = player.car_data.position
            calc = playerHeight.item(2) / CEILING_Z * 100
            if calc < 0:
                return 0
            else:
                reward = np.sqrt(calc)
                if np.isnan(reward) or reward < 0:
                    return 0
                else:
                    # print("Ball touched: " + str(reward))
                    return reward

        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        return 0


class TouchBallReward(RewardFunction):

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not player.ball_touched:
            return -0.007
        else:
            return 20

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        return 0