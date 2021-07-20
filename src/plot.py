import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from environment import Environment


class PlotUtil:
    _results = None
    _rewards = None
    _epsilons = None

    _agent = None
    _best_move_3d = None
    _best_move_2d = None

    @staticmethod
    def plot_best_move(agent: Agent, environment: Environment):
        if PlotUtil._agent is None:
            PlotUtil._agent = plt.figure(figsize=(10, 10))

        if len(environment.dimensions) == 3:
            PlotUtil.plot_best_move_3d(agent, environment)
        elif len(environment.dimensions) == 2:
            PlotUtil.plot_best_move_2d(agent, environment)

    @staticmethod
    def plot_best_move_2d(agent: Agent, environment: Environment):
        if PlotUtil._best_move_2d is None:
            PlotUtil._best_move_2d = PlotUtil._agent.add_subplot(1, 1, 1)

        x = np.arange(0, environment.dimensions[0])
        y = np.arange(0, environment.dimensions[1])
        xs, ys = np.meshgrid(x, y, indexing='ij')

        u = np.zeros(environment.dimensions)
        v = np.zeros(environment.dimensions)

        for _x in x:
            for _y in y:
                location = np.array((_x, _y))
                if environment.is_goal(location):
                    u[_x, _y] = 0
                    v[_x, _y] = 0
                else:
                    state = environment.get_state(location)
                    idx = np.argmax(agent.q_table[state])
                    action = agent.actions[idx]
                    u[_x, _y] = action[0]
                    v[_x, _y] = action[1]

        PlotUtil._best_move_2d.quiver(xs, ys, u, v, pivot='mid')

    @staticmethod
    def plot_best_move_3d(agent: Agent, environment: Environment):
        if PlotUtil._best_move_3d is None:
            PlotUtil._best_move_3d = PlotUtil._agent.add_subplot(1, 1, 1, projection='3d')

        x = np.arange(0, environment.dimensions[0])
        y = np.arange(0, environment.dimensions[1])
        z = np.arange(0, environment.dimensions[2])
        xs, ys, zs = np.meshgrid(x, y, z, indexing='ij')

        u = np.zeros(environment.dimensions)
        v = np.zeros(environment.dimensions)
        w = np.zeros(environment.dimensions)

        for _x in x:
            for _y in y:
                for _z in z:
                    location = np.array((_x, _y, _z))
                    if environment.is_goal(location):
                        u[_x, _y, _z] = 0
                        v[_x, _y, _z] = 0
                        w[_x, _y, _z] = 0
                    else:
                        state = environment.get_state(location)
                        idx = np.argmax(agent.q_table[state])
                        action = agent.actions[idx]
                        u[_x, _y, _z] = action[0]
                        v[_x, _y, _z] = action[1]
                        w[_x, _y, _z] = action[2]

        PlotUtil._best_move_3d.quiver(xs, ys, zs, u, v, w, pivot='middle', length=0.2, arrow_length_ratio=2)

    @staticmethod
    def plot_rewards(rewards, epsilons):
        if PlotUtil._results is None:
            PlotUtil._results = plt.figure(figsize=(10, 10))
            PlotUtil._rewards = PlotUtil._results.add_subplot(1, 1, 1)
            PlotUtil._epsilons = PlotUtil._rewards.twinx()

        PlotUtil._rewards.clear()
        trials = np.arange(0, len(rewards))
        PlotUtil._rewards.plot(trials, rewards)
        PlotUtil._epsilons.plot(trials, epsilons)

    @staticmethod
    def show():
        plt.show()
