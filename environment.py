import numpy as np


# template for 2D environment containing (width, height) and pollution data for each cell. As well as debugging render.
# Contains fnx set_pollution (allowing pollution level to be stored to a cell)
# Contains fnx get_pollution (allowing pollution level to be indexed by cell)
from point3d import Point3d


class Environment:
    # status
    CLEAN = 0
    # POLLUTED = between 0-1
    SOURCE = 1

    def __init__(self, width, height, depth, goal: Point3d):
        self.width = width
        self.height = height
        self.depth = depth
        self.goal = goal
        self.methane_data = np.zeros((width, height, depth))
        self.dissolved_o2_data = np.zeros((width, height, depth))

    def methane_gradient(self):
        return np.gradient(self.methane_data)

    def is_goal(self, x, y, z):
        return x == self.goal.x and y == self.goal.y and z == self.goal.z

    def set_methane(self, location, pollution_level):
        self.methane_data[location.x][location.y][location.z] = pollution_level

    def set_do2(self, x, y, z, do2_level):
        self.dissolved_o2_data[x][y][z] = do2_level

    def get_methane(self, location):
        return self.methane_data[location.x, location.y, location.z]

    def get_do2(self, location):
        return self.dissolved_o2_data[location.x, location.y, location.z]

