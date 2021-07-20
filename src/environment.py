import numpy as np

from utils import manhattan_distance

# todo: add methane (pollution), co2, temp, salinity, conductivity
# todo: change do2 distribution, and maybe update difference between source and methane
# todo: translate temperature to Kelvin
# todo: gas constant coefficients


class Environment:

    def __init__(self, dimensions: tuple, methane: np.ndarray, goal: np.ndarray):
        self.dimensions = dimensions
        # used to adjust location within bounds
        self.cached_max_dimension_index = tuple(map(lambda e: e - 1, self.dimensions))
        self.goal = goal
        self.methane = methane
        # methane_gradient is an array which size is equal to the number of dimensions of the methane data
        # eg. for 2d methane data np.gradient will return 2 `ndarray`,
        # one for each partial derivative along each of the two axis
        # converts the list returned by np.gradient to a `ndarray` for easier slicing
        self.methane_gradient = np.array(np.gradient(self.methane))

    @staticmethod
    def generate_radial_distribution(dimensions: tuple, source: np.ndarray):
        indices = np.indices(dimensions)
        # adds dimensions to the source location so that its rank and the indices' rank are the same
        # this is needed for proper numpy broadcasting when computing values below
        reshaped_source = source.reshape(source.shape + (1,) * (indices.ndim - 1))

        squared_distance = np.sum((indices - reshaped_source) ** 2, axis=0)
        distance_from_source = np.sqrt(squared_distance)

        # computes methane
        methane_levels = 1 / (1 + distance_from_source / 100)

        return Environment(dimensions, methane_levels, source)

    @property
    def state_space_shape(self):
        # each partial derivative is a state parameter
        methane_bucket_count = 3
        return [methane_bucket_count] * len(self.methane_gradient)

    def cardinal_directions(self):
        positive_directions = np.identity(len(self.dimensions), dtype=np.int8)
        return np.concatenate((positive_directions, positive_directions * -1))

    def move_in_bounds(self, location: np.ndarray, delta: np.ndarray):
        location += delta
        return np.clip(location, 0, self.cached_max_dimension_index, out=location)

    def distance_to_goal(self, location: np.ndarray):
        return manhattan_distance(location, self.goal)

    def get_state(self, location: np.ndarray) -> tuple:
        idx = (slice(None),) + tuple(location)
        methane_gradient_at_location = self.methane_gradient[idx]
        return tuple(Environment.binary_state(methane_gradient_at_location))

    def is_goal(self, location: np.ndarray):
        return np.array_equal(self.goal, location)

    def random_location(self):
        return np.random.randint(0, self.dimensions)

    @staticmethod
    def binary_state(state: np.ndarray) -> np.ndarray:
        return np.sign(state).astype(np.int8) + 1
