import random

import numpy as np

from environment import Environment


class Agent:

    # Exploration Factor: 1-epsilon, where epsilon range between 0 to 1,
    # make a random choice epsilon = 1,
    # strictly follow policy epsilon = 0
    # some float combination of the random and policy
    # Discount Factor: Gamma is
    # multiplied by the estimation of the optimal future value.
    # The next reward’s importance is defined by the gamma parameter.
    def __init__(self, env: Environment, learning_rate, discount_rate, epsilon, epsilon_decay):
        self.location = None
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        self.actions = env.cardinal_directions()

        self.q_table = np.zeros(env.state_space_shape + [len(self.actions)])

    def reset(self, environment: Environment):
        self.location = environment.random_location()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < 0.001:
            self.epsilon = 0

    def step(self, environment: Environment):

        state = environment.get_state(self.location)

        # chooses the next action
        action_idx = self.choose_action(state)
        action = self.actions[action_idx]

        # applies the action
        next_state, reward, done = self.act(action, environment)

        goal_reached = (done and environment.is_goal(self.location))

        self.update_q_value(state, next_state, action_idx, goal_reached, reward)

        return reward, done

    def choose_action(self, state):
        # chooses randomly
        if random.random() < self.epsilon:
            return random.randrange(0, len(self.actions))

        # otherwise chooses using the q table
        return np.argmax(self.q_table[state])

    def act(self, delta, environment: Environment):
        self.location = environment.move_in_bounds(self.location, delta)

        next_state = environment.get_state(self.location)
        reward = -1
        done = False

        if environment.is_goal(self.location):
            done = True
            reward = 100

        return next_state, reward, done

    def update_q_value(self, previous_state, current_state, action_idx, goal_reached, reward):

        # q_values for the previous state
        q_cell = self.q_table[previous_state]

        # if the goal was reached, directly updates the q_value to the reward
        if goal_reached:
            q_cell[action_idx] = reward
            return

        # otherwise updates q value using learning rate and discounted utility
        current_state_actions = self.q_table[current_state]
        max_utility_from_current_state = max(current_state_actions)

        q_old = q_cell[action_idx]
        q_new = (reward + self.discount_rate * max_utility_from_current_state)

        q_cell[action_idx] = (1 - self.learning_rate) * q_old + self.learning_rate * q_new
