class Point3d:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def clone(self):
        return Point3d(self.x, self.y, self.z)

    def add(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def manhattan_distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def to_string(self):
        return '(%s, %s, %s)' % (self.x, self.y, self.z)

    def equals(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
