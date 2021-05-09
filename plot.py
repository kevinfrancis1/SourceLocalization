import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class PlotUtil:

    def __init__(self, width, height, depth):
        self.fig = plt.figure(figsize=(16, 8))
        self.pollution = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.u = self.fig.add_subplot(1, 2, 2, projection='3d')

        x = np.arange(0, width)
        y = np.arange(0, height)
        z = np.arange(0, depth)

        self.xs, self.ys, self.zs = np.meshgrid(x, y, z)

    def init_q_table_plot(self):
        # Put in its own setup function
        self.u.set_xlabel('X Label')
        self.u.set_ylabel('Y Label')
        self.u.set_zlabel('Z Label')

    def init_pollution_plot(self):
        self.pollution.set_xlabel('X Label')
        self.pollution.set_ylabel('Y Label')
        self.pollution.set_zlabel('Z Label')

    def render_utility(self, utility, stop=False):
        # ax is a global
        self.u.clear()
        self.u.set(title="Utility Table", xlabel="x", ylabel='y', zlabel="z")

        min_u = utility.min()
        max_u = utility.max()

        if max_u - min_u == 0:
            return

        normalized_utilities = (utility - min_u) / (max_u - min_u)
        self.u.scatter(self.xs, self.ys, self.zs, s=normalized_utilities * 50)

        if stop:
            plt.show()

    def render_pollution_grid(self, pollution):
        self.pollution.clear()
        self.pollution.set(title="Pollution Grid", xlabel="x", ylabel='y', zlabel="z")
        min_pollution = pollution.min()
        max_pollution = pollution.max()

        normalize = cm.colors.Normalize(vmin=min_pollution, vmax=max_pollution)
        scalar_map = cm.ScalarMappable(norm=normalize, cmap=cm.viridis)
        flat_pollution = pollution.ravel()

        sizes = (pollution - min_pollution) / (max_pollution - min_pollution) * 100
        colors = scalar_map.to_rgba(flat_pollution)

        self.pollution.scatter(self.xs, self.ys, self.zs, s=sizes, c=colors)

        # todo:
        # for an xyz color this point between [0 1] based on the normalzied self.pollution_data

    # visualizes agent in space in the Run window.
    def render_agent(self, agent, old=False, pause=0.01):
        size = 200
        color = 'gray' if old else 'red'

        # self.ax.clear()
        self.pollution.scatter(agent.x, agent.y, agent.z, s=size, c=color, marker="d")
        if pause > 0:
            plt.pause(pause)

        # ax.cla()
        # color_map1 = plt.get_cmap('YlGnBu')
        # color_map2 = plt.get_cmap('RdBu')

        # x = np.arange(0, self.width)
        # # #x = x.flatten()
        # y = np.arange(0, self.height)
        # # #y = y.flatten()
        # z = np.arange(0, self.depth)
        # z = self.pollution_data[:, 2]
        # #z = z.flatten()
        # X, Y = np.meshgrid(x, y)
        # print(x)
        # print(y)
        # ax.plot_surface(X, Y, z, cmap=cm.viridis,
        #                    linewidth=0, antialiased=False)

        # ax.matshow(self.dissolved_o2_data, cmap=color_map2, alpha=0.0)
        # ax.set_box_aspect((agent.x), (agent.y), (agent.z))
