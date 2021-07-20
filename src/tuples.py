import math

import numpy as np


class Tuple:

    @property
    def tuple(self):
        pass

    def add(self, other):
        pass

    def distance_to(self, other):
        pass

    def equals(self, other):
        pass


class Tuple2d(Tuple):

    def __init__(self, e1, e2):
        self.a = np.array([e1, e2])

    @property
    def x(self):
        return self.a[0]

    @property
    def y(self):
        return self.a[1]

    @property
    def width(self):
        return self.a[0]

    @property
    def height(self):
        return self.a[1]

    @property
    def tuple(self):
        return self.a[0], self.a[1]

    def add(self, other):
        self.a += other.a

    def equals(self, other):
        return np.array_equal(self.a, other.a)

    def distance_to(self, other):
        return np.sqrt(np.sum((self.a - other.a)**2))

    def manhattan_distance_to(self, other):
        return np.sum(abs(self.a - other.a))
    #
    # def to_string(self):
    #     return '(%s, %s)' % (self.x, self.y)
#
#
# class Tuple3d(Tuple):
#
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z
#
#     def to_tuple(self):
#         return self.x, self.y, self.z
#
#     def equals(self, other):
#         return self.x == other.x and self.y == other.y and self.z == other.z
#
#     def distance_to(self, other):
#         return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
#
#     def clone(self):
#         return Tuple3d(self.x, self.y, self.z)
#
#     def add(self, other):
#         self.x += other.x
#         self.y += other.y
#         self.z += other.z
#
#     def to_string(self):
#         return '(%s, %s, %s)' % (self.x, self.y, self.z)
#
#