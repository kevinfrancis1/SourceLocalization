#!/usr/bin/python3
# expand the world states
# coordinate grid 10x10
# cells contain percentage 0-100 pollution
import random
import math
import time

from environment import Environment
from agent import Agent
from plot import PlotUtil
from point3d import Point3d

# scoring
from utils import file_print, create_log

performance_measure = []

# utility_measure = []


def get_logarithmic_methane_boundary(depth):
    x = (-depth+12)**7
    offset = 19
    return - math.log(x) + offset


def get_linear_methane_boundary(depth):
    return (depth + 1) * 0.75


def get_methane_for_depth(depth):
    return -0.07*depth+1


def get_methane(cell, source):
    cell_in_source_reference_frame = Point3d(cell.x - source.x, cell.y - source.y, cell.z)

    planar_distance_to_source = math.sqrt(cell_in_source_reference_frame.x**2 + cell_in_source_reference_frame.y**2)
    # methane_boundary_distance = get_linear_methane_boundary(cell_in_source_reference_frame.z)
    methane_boundary_distance = get_logarithmic_methane_boundary(cell_in_source_reference_frame.z)
    methane_ratio = planar_distance_to_source / methane_boundary_distance
    max_methane_at_z = get_methane_for_depth(cell.z)
    methane = (1 - methane_ratio) * max_methane_at_z
    # print(max(methane, 0))
    return max(methane, 0)


# Generates "world" setting size in 3D, location of the source and "pollution" in each cell
# gradient/noise level of pollution smaller gradient -> more diffuse. More noise -> more stochastic
def generate_environment(width: int, height: int, depth: int, source: Point3d, grad=0.1, noise_amplitude=0.0):
    environment = Environment(width, height, depth, source)

    # Sets the pollution diffusion from source to be no-noise or noisy

    # todo: add methane (pollution), co2, temp, salinity, conductivity
    # todo: change do2 distribution, and maybe update difference between source and methane
    # todo: translate temperature to Kelvin
    # todo: gas constant coefficients
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                cell = Point3d(x, y, z)
                methane = get_methane(cell, source)
                environment.set_methane(cell, methane)
    return environment


# computes cost of current location and provides a performance measure as a function of cost. Can be used later for
# probabilities, planning and learning.
# def compute_utility(agent):
#     utility = agent.utility_table(agent.location)
#     print(utility)
#     return utility


def compute_cost(agent, environment):
    status = environment.get_methane(agent.location)
    return 1 - status


def compute_performance(cost):
    return 1 / (cost + 1)
    # if status == CLEAN:
    #     return 10
    # if status == POLLUTED:
    #     return 5
    # if status == SOURCE:
    #     return 0


# Agent chooses an action, Action is applied to the environment by moving the agent
# a way of keeping track of time and status in world in order to find the SOURCE
# change in the environment = location + action
def program(environment, agent, plot):
    cost = 0
    i = 0

    # sets "lifetime" of agent in world, render environment
    for i in range(10000):
        #plot.render_agent(agent, old=False, pause=0.01)

        # outputs data as a comma separated array
        # file_print([i, agent.last_pollution(), cost])
        cost += compute_cost(agent, environment)

        # -- Step choose an action using the provided agent.
        action = agent.choose_action(environment)
        if action == agent.NONE:
            break
        # based on the action returned by the agent for the given state (environment and current location)
        # apply the action
        agent.act(action, environment)

        # print(initial_location, environment_status, location)
        # print(score)
        # -- End Step

        # plot.render_pollution_grid(environment.methane_data)
        # plot.render_agent(agent)
    return compute_performance(cost), i


if __name__ == "__main__":
    # env = generate_environment(20, 10, 0.35)
    # print(env)
    width = 12
    height = 12
    depth = 12
    source_location = Point3d(5, 5, 0)
    agent_starting_location = Point3d(0, 0, 0)

    plot = PlotUtil(width, height, depth)
    water = generate_environment(width, height, depth, source_location, 1/21, 0.0)
    bob = Agent(agent_starting_location, width, height, depth, 0.95, 1)

    # render the pollution grid in 3D
    # plot.render_pollution_grid(water.methane_data, True)

    log_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + '-12x12x12_5x5x5_train_stochastic'
    create_log(log_name)
    totst = []
    totstnorm = []
    totavg = []
    totavgnorm = []
    k = 1
    # while k < 11:
    # print("Starting run " + str(k))

    for run_index in range(100):
        print("Finished run %s" % (run_index, ))
        distance_to_source = bob.distance_to(water.goal)
        epsilon = bob.epsilon
        perf, total_step = program(water, bob, plot)
        bob.update_q_states()
        plot.render_utility(bob.q_table)

        bob.reset()
        # print(run_index, total_step, distance_to_source, total_step / distance_to_source)

        performance_measure.append(perf)
        # prints to csv size, source location, distance, gradient, noise, steps and performance
        file_print([run_index, epsilon, total_step, distance_to_source, (total_step + 1) / (distance_to_source + 1)])
        # totst.append(total_step)
        # totstnorm.append((total_step + 1) / (distance_to_source + 1))
        # k += 1
        # totavgnorm.append(sum(totstnorm)/len(totstnorm))
        # totavg.append(sum(totst)/len(totst))
    # print("Average of average steps: " + str(sum(totavg)/len(totavg)))
    # print("Average of averages normalized steps: " + str(sum(totavgnorm)/len(totavgnorm)))
    # file_print(bob.utility_table)
        bob.print_utilities()

    log_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + '-12x12x12_5x5x5_test_stochastic'
    create_log(log_name)
    # water = generate_environment(width, height, depth, 5, 7, 6, 1/21, 0.0)
    bob.epsilon = 0.0
    for run_index in range(0):
        # print("Now Testing Agent")
        print("Finished run %s" % (run_index, ))
        distance_to_source = bob.distance_to(water.goal)
        epsilon = bob.epsilon

        perf, total_step = program(water, bob, plot)
        # bob.update_utilities()
        # plot.render_utility(bob.utility_table)

        bob.reset()
        # print(run_index, total_step, distance_to_source, total_step / distance_to_source)

        performance_measure.append(perf)
        # prints to csv size, source location, distance, gradient, noise, steps and performance
        file_print([run_index, epsilon, total_step, distance_to_source, (total_step + 1) / (distance_to_source + 1)])

    # file_print(bob.utility_table)
        # bob.print_utilities()

    plot.render_utility(bob.q_table, stop=True)

    # performance_measure.append(program(0, generate_environment(10, 7), reflex_agent()));

    # performance_measure.append(program(0, [0, 0.5, 1], reflex_agent))

    # performance_measure.append(program(1, [POLLUTED, CLEAN, POLLUTED, SOURCE], reflex_agent))

    # performance_measure.append(program(0, [SOURCE, CLEAN, POLLUTED], reflex_agent))

    # performance_measure.append(program(0, [CLEAN, CLEAN, POLLUTED, SOURCE, POLLUTED, CLEAN], reflex_agent))

    # print('Total steps = ', total_step)
    # print("Performance = ", performance_measure)
    # print("Global Performance = ", mean(performance_measure))
