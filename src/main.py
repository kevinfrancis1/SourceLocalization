#!/usr/bin/python3

import numpy as np

from plot import PlotUtil
from environment import Environment
from agent import Agent


def run():
    dimensions = (21, 13, 8)
    source_location = np.array([5, 3, 2])
    environment = Environment.generate_radial_distribution(dimensions, source_location)
    agent = Agent(environment, 0.1, 0.9, 1, 0.995)

    # utils.print_methane(environment)
    # graphs.plot_methane(environment.methane)
    # graphs.plot_methane_gradient(environment.methane_gradient, -1)
    # graphs.plot_methane_gradient(environment.methane_gradient, -1)

    run_epoch(environment, agent, 1000)
    PlotUtil.plot_best_move(agent, environment)
    PlotUtil.show()
    # utils.print_best_move(environment, agent)
    # utils.print_bucket(environment, agent)


def run_epoch(environment: Environment, agent: Agent, trial_count):
    rewards = []
    epsilons = []

    for trial_idx in range(trial_count):
        reward, step = trial(environment, agent, 200)
        print(f"Finished trial {trial_idx} on step {step} - epsilon {agent.epsilon:.3f}, adjusted reward: {reward}")
        rewards.append(reward)
        epsilons.append(agent.epsilon)

    PlotUtil.plot_rewards(rewards, epsilons)


def trial(environment: Environment, agent: Agent, steps):

    agent.reset(environment)
    manhattan_distance_from_source = environment.distance_to_goal(agent.location)
    reward = 0
    i = 0

    # sets "lifetime" of agent in world
    for i in range(steps):
        current_reward, done = agent.step(environment)
        reward += current_reward
        if done:
            break
        # graphs.plot_agent_and_environment(agent, environment)

    agent.decay_epsilon()
    return reward + manhattan_distance_from_source, i


if __name__ == "__main__":
    run()
