
DEBUG = False

file = None


def create_log(name):
    global file
    file = open('data/perf/size/' + name + '-data.csv', 'w')


def file_print(data):
    print(', '.join(map(str, data)), file=file)


def debug_print(msg):
    if DEBUG:
        print(msg)
