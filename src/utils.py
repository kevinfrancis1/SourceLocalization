import numpy as np

DEBUG = False

file = None


def euclidean_distance(a1: np.ndarray, a2: np.ndarray):
    return np.sqrt(np.sum((a1 - a2) ** 2))


def manhattan_distance(a1: np.ndarray, a2: np.ndarray):
    return np.sum(abs(a1 - a2))


def create_log(name):
    global file
    file = open('data/perf/size/' + name + '-data.csv', 'w')


def file_print(data):
    print(', '.join(map(str, data)), file=file)


def debug_print(msg):
    if DEBUG:
        print(msg)
