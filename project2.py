#!/usr/bin/python3
# expand the world states
# coordinate grid 10x10
# cells contain percentage 0-100 pollution
import random
import numpy as np
import time

from environment import Environment
from agent import Agent
from plot import PlotUtil

# scoring
from utils import file_print, create_log

performance_measure = []


# utility_measure = []

# Generates "world" setting size in 3D, location of the source and "pollution" in each cell
# gradient/noise level of pollution smaller gradient -> more diffuse. More noise -> more stochastic
def generate_environment(width: int, height: int, depth: int, source_x: int, source_y: int, source_z: int, grad=0.1,
                         noise_amplitude=0.0):
    environment = Environment(width, height, depth, (source_x, source_y, source_z))

    # Sets the pollution diffusion from source to be no-noise or noisy
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                delta_x = source_x - x
                delta_y = source_y - y
                delta_z = source_z - z
                # formula for distance b/w 2 points
                distance = np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
                # establishes difference in pollution level from cell to cell
                distance_gradient = distance * grad
                pollution_level = 1 - distance_gradient
                # print(pollution_level)
                do2_level = 0 + .60 * distance_gradient

                if pollution_level < 1:
                    # min prevents noise level from making pollution > 1. prevents multiple false sources
                    local_noise = min(noise_amplitude, distance_gradient)
                    noise_pollution = random.random() * local_noise
                    final_pollution_level = max(0.0, pollution_level + noise_pollution)
                    # print(final_pollution_level)
                    environment.set_pollution(x, y, z, final_pollution_level)
                    final_do2_level = min(1.0, do2_level)
                    environment.set_do2(x, y, z, final_do2_level)
                else:
                    noise_pollution = 0
                    # max prevents pollution level from generating negative values
                    final_pollution_level = max(0.0, pollution_level + noise_pollution)
                    environment.set_pollution(x, y, z, final_pollution_level)
                    final_do2_level = min(1.0, do2_level)
                    environment.set_do2(x, y, z, final_do2_level)
    return environment


# Makes a decision about the action to do
# Agent knows its location, perceives its status and can "recall" the status of its previous location and the action
# it took prior
# rules apply hill climbing global maximum if previous action < current state continue If previous > current turn around
# expand memory of agent to include last 1-4 moves add a threshold data to reference2-4 nodes
# implement sim anneal when threshold is not increased over time then random


# computes cost of current location and provides a performance measure as a function of cost. Can be used later for
# probabilities, planning and learning.
# def compute_utility(agent):
#     utility = agent.utility_table(agent.location)
#     print(utility)
#     return utility


def compute_cost(agent, environment):
    status = environment.get_pollution(agent.x, agent.y, agent.z)
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

    # sets "lifetime" of agent in world render environment
    for i in range(1000):
        #plot.render_agent(agent, old=True, pause=0)

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

        #plot.render_pollution_grid(environment.pollution_data)
        #plot.render_agent(agent)
    return compute_performance(cost), i


if __name__ == "__main__":
    # env = generate_environment(20, 10, 0.35)
    # print(env)
    width = 12
    height = 12
    depth = 12

    plot = PlotUtil(width, height, depth)
    water = generate_environment(width, height, depth, 5, 5, 5, 1/21, 0.0)
    bob = Agent(1, 1, 1, width, height, depth, 0.95, 1)


    # render the pollution grid in 3D
    #plot.render_pollution_grid(water.pollution_data)

    log_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + '-12x12x12_5x5x5_train_stochastic'
    create_log(log_name)
    totst = []
    totstnorm = []
    totavg = []
    totavgnorm = []
    k = 1
    #while k < 11:
        #print("Starting run " + str(k))

    for run_index in range(2500):
        print("Finished run %s" % (run_index, ))
        distance_to_source = bob.distance_to(water.goal)
        epsilon = bob.epsilon
        perf, total_step = program(water, bob, plot)
        bob.update_utilities()
        #plot.render_utility(bob.utility_table)

        bob.reset()
        # print(run_index, total_step, distance_to_source, total_step / distance_to_source)

        performance_measure.append(perf)
        # prints to csv size, source location, distance, gradient, noise, steps and performance
        file_print([run_index, epsilon, total_step, distance_to_source, (total_step + 1) / (distance_to_source + 1)])
        #totst.append(total_step)
        #totstnorm.append((total_step + 1) / (distance_to_source + 1))
        #k += 1
        #totavgnorm.append(sum(totstnorm)/len(totstnorm))
        #totavg.append(sum(totst)/len(totst))
    #print("Average of average steps: " + str(sum(totavg)/len(totavg)))
    #print("Average of averages normalized steps: " + str(sum(totavgnorm)/len(totavgnorm)))
    #file_print(bob.utility_table)
        # bob.print_utilities()

    log_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + '-12x12x12_5x5x5_test_stochastic'
    create_log(log_name)
    #water = generate_environment(width, height, depth, 5, 7, 6, 1/21, 0.0)
    bob.epsilon = 0.0
    for run_index in range(2500):
        #print("Now Testing Agent")
        print("Finished run %s" % (run_index, ))
        distance_to_source = bob.distance_to(water.goal)
        epsilon = bob.epsilon

        perf, total_step = program(water, bob, plot)
        #bob.update_utilities()
        #plot.render_utility(bob.utility_table)

        bob.reset()
        # print(run_index, total_step, distance_to_source, total_step / distance_to_source)

        performance_measure.append(perf)
        # prints to csv size, source location, distance, gradient, noise, steps and performance
        file_print([run_index, epsilon, total_step, distance_to_source, (total_step + 1) / (distance_to_source + 1)])

    #file_print(bob.utility_table)
        # bob.print_utilities()

    #plot.render_utility(bob.utility_table, stop=True)

    # performance_measure.append(program(0, generate_environment(10, 7), reflex_agent()));

    # performance_measure.append(program(0, [0, 0.5, 1], reflex_agent))

    # performance_measure.append(program(1, [POLLUTED, CLEAN, POLLUTED, SOURCE], reflex_agent))

    # performance_measure.append(program(0, [SOURCE, CLEAN, POLLUTED], reflex_agent))

    # performance_measure.append(program(0, [CLEAN, CLEAN, POLLUTED, SOURCE, POLLUTED, CLEAN], reflex_agent))

    # print('Total steps = ', total_step)
    # print("Performance = ", performance_measure)
    # print("Global Performance = ", mean(performance_measure))
