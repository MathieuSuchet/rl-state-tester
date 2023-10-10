from abc import ABC, abstractmethod
from typing import List, Union, Dict, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from rlgym_sim.gym import Gym
from rlgym_sim.utils.common_values import BACK_WALL_Y, SIDE_WALL_X, CEILING_Z
from rlgym_sim.utils.gamestates import GameState
from stable_baselines3 import PPO

from rl_state_tester.global_harvesters.standalone_runner import StandaloneRunner

CORNERS = np.array([
    np.array([-SIDE_WALL_X, BACK_WALL_Y, 0]),
    np.array([-SIDE_WALL_X, -BACK_WALL_Y, 0]),
    np.array([SIDE_WALL_X, BACK_WALL_Y, 0]),
    np.array([SIDE_WALL_X, -BACK_WALL_Y, 0]),
    np.array([-SIDE_WALL_X, BACK_WALL_Y, CEILING_Z]),
    np.array([-SIDE_WALL_X, -BACK_WALL_Y, CEILING_Z]),
    np.array([SIDE_WALL_X, BACK_WALL_Y, CEILING_Z]),
    np.array([SIDE_WALL_X, -BACK_WALL_Y, CEILING_Z])
])

GOAL_X = 893

GOAL_ORANGE = np.array([
    [-GOAL_X, BACK_WALL_Y],
    [GOAL_X, BACK_WALL_Y],
])

GOAL_BLUE = np.array([
    [-GOAL_X, -BACK_WALL_Y],
    [GOAL_X, -BACK_WALL_Y],
])


class PlayerStats(NamedTuple):
    avg: float
    median: float
    std: float
    var: float

    def __str__(self):
        return f"\t\tAverage: {self.avg}\n" \
               f"\t\tMedian: {self.median}\n" \
               f"\t\tStd: {self.std}\n" \
               f"\t\tVar: {self.var}\n"


def calculate_stats_from_players(players):
    all_stats = []

    # For each player
    for i in range(players.shape[1]):
        player = players[:, i, :]

        for axis in range(player.shape[-1]):
            all_stats.append(
                PlayerStats(
                    np.mean(player[:, axis]),
                    np.median(player[:, axis]),
                    np.std(player[:, axis]),
                    np.var(player[:, axis])
                )
            )

    return all_stats


class StatHarvester(StandaloneRunner, ABC):

    def __init__(self, env: Gym, agent: PPO, rendered: bool, deterministic: bool):
        super().__init__(env, agent, rendered, deterministic)
        self.data = []
        self.n_episodes = -1

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        state: GameState = info["state"]
        self.data.append(np.array(self._get_data(state)))

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        self.n_episodes += 1

    def _on_close(self, *args, **kwargs):
        self.data = np.array(self.data)

    @abstractmethod
    def _get_data(self, state: GameState):
        pass

    @abstractmethod
    def customize_plot(self, ax: Axes):
        pass


class MultiStatHarvester(StatHarvester):
    def _get_data(self, state: GameState):
        pass

    def __init__(self, env: Gym, agent: PPO, rendered: bool, deterministic: bool, harvesters: List[StatHarvester]):
        super().__init__(env, agent, rendered, deterministic)

        self.harvesters = harvesters

    def _on_step(self, obs: np.array, action: np.array, reward: List[Union[float, int]], terminal: Union[List[bool], bool],
                 info: Dict[str, object], *args, **kwargs):
        for harvester in self.harvesters:
            harvester.on_step(obs, action, reward, terminal, info, args, kwargs)

    def _on_reset(self, obs: np.array, info: Dict[str, object], *args, **kwargs):
        for harvester in self.harvesters:
            harvester.on_reset(obs, info, args, kwargs)

    def _on_close(self, *args, **kwargs):
        for harvester in self.harvesters:
            harvester.on_close(args, kwargs)

        self.customize_plot()

    def customize_plot(self, **kwargs):
        n_harvesters = len(self.harvesters)

        plt.style.use("dark_background")
        fig = plt.figure()
        for i, harvester in enumerate(self.harvesters):
            print(100 + 10 * n_harvesters + (i + 1))
            ax = fig.add_subplot(100 + 10 * n_harvesters + (i + 1), projection="3d")
            harvester.customize_plot(ax)
            ax.legend()

        plt.legend()
        plt.show()


class PlayerPositionHarvester(StatHarvester):
    def _get_data(self, state: GameState):
        return [player.car_data.position for player in state.players]

    def customize_plot(self, ax: Axes):
        ax.set_title("Cars position")
        ax.set_xlabel("Y axis")
        ax.set_ylabel("X axis")
        ax.set_zlabel("Height")

        ax.scatter(CORNERS[:, 1], CORNERS[:, 0], CORNERS[:, 2], c="yellow")

        colors = []
        legends = []

        nb_players = self.data.shape[1]

        mid = nb_players // 2
        print(mid)

        for j in range(mid):
            print(j)
            if j == 0:
                colors.append("green")
                legends.append("Your player (1 Blue)")
            else:
                colors.append("blue")
                legends.append(f"{j + 1} Blue")

        for j in range(mid, nb_players):
            print(j)
            colors.append("orange")
            legends.append(f"{j - mid + 1} Orange")

        for i in range(nb_players):
            temp = self.data[:, i, :]
            ax.scatter(temp[:, 1], temp[:, 0], temp[:, 2], c=colors[i], label=legends[i])

        ax.plot(GOAL_ORANGE[:, 1], GOAL_ORANGE[:, 0], color="red")
        ax.plot(GOAL_BLUE[:, 1], GOAL_BLUE[:, 0], color="red")


class FlattenedStatHarvester(StatHarvester, ABC):
    def customize_plot(self, ax: Axes):
        n_axes = self.data.shape[-1]
        n_steps = range(self.data.shape[0])

        fig, axes = plt.subplots(ncols=n_axes)

        for i in range(n_axes):
            colors = []
            legends = []

            nb_players = self.data.shape[1]
            mid = nb_players // 2
            print(mid)

            for j in range(mid):
                if j == 0:
                    colors.append("green")
                    legends.append("Your player (1 Blue)")
                else:
                    colors.append("blue")
                    legends.append(f"{j + 1} Blue")

            for j in range(mid, nb_players):
                colors.append("orange")
                legends.append(f"{j - mid + 1} Orange")

            for k in range(nb_players):
                temp = self.data[:, k, :]

                print(temp[:, i])
                axes[i].scatter(n_steps, temp[:, i], c=colors[k], label=legends[k])

        fig.show()


class FlattenedPlayerPositionHarvest(FlattenedStatHarvester):
    def _get_data(self, state: GameState):
        return [player.car_data.position for player in state.players]


class PlayerLinearVelHarvester(StatHarvester):
    def _get_data(self, state: GameState):
        return [player.car_data.linear_velocity for player in state.players]

    def customize_plot(self, ax: Axes):
        ax.set_title("Cars velocity")
        ax.set_xlabel("Y axis")
        ax.set_ylabel("X axis")
        ax.set_zlabel("Steps")

        colors = []
        legends = []

        nb_players = self.data.shape[1]
        mid = nb_players // 2
        print(mid)

        for j in range(mid):
            if j == 0:
                colors.append("green")
                legends.append("Your player (1 Blue)")
            else:
                colors.append("blue")
                legends.append(f"{j + 1} Blue")

        for j in range(mid, nb_players):
            colors.append("orange")
            legends.append(f"{j - mid + 1} Orange")

        n_steps = range(self.data[:, 0, :].shape[0])

        for i in range(nb_players):
            temp = self.data[:, i, :]
            ax.scatter(temp[:, 1], temp[:, 0], n_steps, c=colors[i], label=legends[i])


class BallPositionHarvester(StatHarvester):

    def _get_data(self, state: GameState):
        return state.ball.position

    def customize_plot(self, ax: Axes):
        ax.set_title("Ball position")
        ax.set_xlabel("Y axis")
        ax.set_ylabel("X axis")
        ax.set_zlabel("Height")

        ax.plot(GOAL_ORANGE[:, 1], GOAL_ORANGE[:, 0], color="red")
        ax.plot(GOAL_BLUE[:, 1], GOAL_BLUE[:, 0], color="red")

        ax.scatter(CORNERS[:, 1], CORNERS[:, 0], CORNERS[:, 2], c="yellow")
        ax.scatter(self.data[:, 1], self.data[:, 0], self.data[:, 2], c="black",
                   label="Ball")
