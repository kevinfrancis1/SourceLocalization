import numpy as np


# template for 2D environment containing (width, height) and pollution data for each cell. As well as debugging render.
# Contains fnx set_pollution (allowing pollution level to be stored to a cell)
# Contains fnx get_pollution (allowing pollution level to be indexed by cell)
class Environment:
    # status
    CLEAN = 0
    # POLLUTED = between 0-1
    SOURCE = 1

    def __init__(self, width, height, depth, goal):
        self.width = width
        self.height = height
        self.depth = depth
        self.goal = goal
        self.pollution_data = np.zeros((width, height, depth))
        self.dissolved_o2_data = np.zeros((width, height, depth))

    def is_goal(self, x, y, z):
        return x == self.goal[0] and y == self.goal[1] and z == self.goal[2]

    def set_pollution(self, x, y, z, pollution_level):
        self.pollution_data[x][y][z] = pollution_level

    def set_do2(self, x, y, z, do2_level):
        self.dissolved_o2_data[x][y][z] = do2_level

    def get_pollution(self, x, y, z):
        # print(x, y, z)
        return self.pollution_data[x][y][z]

    def get_do2(self, x, y, z):
        return self.dissolved_o2_data[x][y][z]

