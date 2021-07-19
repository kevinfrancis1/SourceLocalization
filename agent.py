import random
import sys
from environment import Environment
from point3d import Point3d
from utils import debug_print
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REWARD = 100


# Template for the agent containing agents x and y coordinate within the world.
# Contains fnx initializing the location of the agent within the world
# fnx is_at provides agent location at specific step in the world
# fnx move updates agent location based off of change in x and y
# fnx move_: e,w,n,s provide changes to the appropriate coordinate to move N, S, E, W.  origin at bottom left


class Agent:
    NONE = 'none'
    POSX = Point3d(1, 0, 0)
    NEGX = Point3d(-1, 0, 0)
    NEGY = Point3d(0, -1, 0)
    POSY = Point3d(0, 1, 0)
    POSZ = Point3d(0, 0, 1)
    NEGZ = Point3d(0, 0, -1)

    MOVES = [NEGY, POSY, POSX, NEGX, POSZ, NEGZ]

    def __init__(self, location: Point3d, space_width, space_height, space_depth, epsilon_decay=0.95, epsilon=1):
        self.location = location

        self.width = space_width
        self.height = space_height
        self.depth = space_depth

        self.epsilon_decay = epsilon_decay

        # Exploration Factor: 1-epsilon, where epsilon range between 0 to 1,
        # make a random choice epsilon = 1,
        # strictly follow policy epsilon = 0
        # some float combination of the random and policy

        self.epsilon = epsilon

        # Discount Factor: Gamma is
        # multiplied by the estimation of the optimal future value.
        # The next reward’s importance is defined by the gamma parameter.
        self.gamma = 0.9

        # Initializes the previous_action to NONE. keeps track of the action taken prior to get to current tile
        self.previous_action = Agent.NONE
        # Initializes the previous_state to 0. keeps track of the status in the tile visited prior
        # location, pollution, DO2, reward
        # Make Dict
        self.experiences = []
        # Make 3D numpy array
        # utility table array of arrays methane value of d10 and methane gradient (x,y,z) of d12 and moves of d6
        # todo add O2, O2gradient and salinity once they have been added to environment.
        self.q_table = np.zeros((10, 12, 12, 12, 6))

    def distance_to(self, point):
        return self.location.manhattan_distance(point)

    def last_pollution(self):
        if len(self.experiences) == 0:
            return 0
        return self.experiences[len(self.experiences) - 1][0]

    def last_gradient(self):
        if len(self.experiences) == 0:
            return 0
        return self.experiences[len(self.experiences) - 1][1]

    def last_do2(self):
        if len(self.experiences) == 0:
            return 0
        return self.experiences[len(self.experiences) - 1][2]

    def potential_moves(self):
        moves = []
        if self.location.x > 0:
            moves.append(Agent.NEGX)

        if self.location.x < self.width - 1:
            moves.append(Agent.POSX)

        if self.location.y > 0:
            moves.append(Agent.NEGY)

        if self.location.y < self.height - 1:
            moves.append(Agent.POSY)

        if self.location.z > 0:
            moves.append(Agent.NEGZ)

        if self.location.z < self.depth - 1:
            moves.append(Agent.POSZ)

        return moves

    def choose_action(self, environment: Environment):

        # perceive environment status based on location
        pollution = environment.get_methane(self.location)
        do2 = environment.get_do2(self.location)

        # if agent is at source it will do nothing
        if pollution == Environment.SOURCE:
            debug_print('source found at, %s. Pollution: %s do2: %s' % (self.location.to_string(), pollution, do2))
            print('Found Source')
            return Agent.NONE

        # Error checking
        if self.location.x < 0 or self.location.y < 0 or self.location.z < 0:
            print('ERROR: Agent is outside of bounds, exiting')
            sys.exit()

        # prevents agent from getting stuck in initial "previous_action"
        # if self.previous_action == Agent.NONE:
        #   return random.choice((Agent.NEGX, Agent.POSX, Agent.NEGY, Agent.POSY, Agent.NEGZ, Agent.POSZ))

        # exploration using epsilon decay
        # encourages agent to explore early in the process
        moves = self.potential_moves()

        # Exploration
        if random.random() < self.epsilon:
            action = random.choice(moves)
            debug_print('Moving %s from %s. Pollution: %s do2: %s'
                        % (action, self.location.to_string(), pollution, do2))
            return action

        # Exploitation of Action Policy
        # exploit
        best_utility = None
        best_move = None
        # updates action in q table
        # todo check to make sure this is updating the action in the q-table idx(4)
        for move in moves:
            debug_print('Checking move %s.' % (move.to_string()))
            next_local = self.location.clone()
            next_local.add(move)
            # todo this doesn't work because it only points to the action slice, not the action paired with states
            utility = self.q_table[4]
            if best_utility is None or utility > best_utility:
                best_utility = utility
                best_move = move

        debug_print('choose based on utility')
        return best_move

    # todo bucketize each parameter. Bucketize function added to utils file, needs to be implemented in the utility update.
    # todo how to find the expected future reward.

    def update_q_states(self):
        # todo re-write function based on bellman eq.
    def update_q_states(self):
        exp_count = len(self.experiences)
        trial_u = [0] * exp_count
        for exp_idx in range(exp_count, 0, -1):
            experience = self.experiences[exp_idx - 1]
            exp_reward = experience[3]
            for r_idx in range(exp_idx, 0, -1):
                trial_u[r_idx - 1] += exp_reward
                exp_reward *= self.gamma

        # alpha is 0.5 (utility+trialu/2). adjust alpha by changing denominator
        for exp_idx in range(exp_count):
            experience = self.experiences[exp_idx]
            pollution = experience[0]
            methane_gradient = experience[1]
            current_utility = self.q_table[0, 1, 2, 3, 4]
            self.q_table[0, 1, 2, 3, 4] = (current_utility + trial_u[exp_idx]) / 2

    def print_utilities(self):
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    print('{:.2f} '.format(self.q_table[x, y, z]), end='')
            print()

    def reset(self):
        # todo once total steps are normalized in the project.py reset to random positions (this is old... maybe
        #  don't do this)
        self.epsilon *= self.epsilon_decay
        self.location.x = random.randint(0, self.width - 1)
        self.location.y = random.randint(0, self.height - 1)
        self.location.z = random.randint(0, self.depth - 1)
        self.experiences.clear()

    # todo update reward based on underlying phenomena
    # def reward(self, environment):
    #     reward = 0
    #     if environment.is_goal(self.x, self.y):
    #         reward = REWARD
    #     return reward

    # Pollution reward
    def reward(self, environment):
        if environment.get_methane(self.location) == Environment.SOURCE:
            return 100
        return 0

    def act(self, action, environment: Environment):
        # allows agent to know action and status of prior environment
        pollution = environment.get_methane(self.location)

        do2 = environment.get_do2(self.location)

        self.previous_action = action

        # if action = NONE that means agent is at the source and should stop.
        if action == Agent.NONE:
            return

        self.move(action)

        reward = self.reward(environment)
        # Experiences hold methane gradient, do2 and reward
        self.experiences.append(
            (
                pollution,
                environment.methane_gradient(),
                do2,
                reward

            )
        )

    def move(self, delta: Point3d):
        self.location.add(delta)

    def set_utility(self, location, utility):
        self.q_table[location] = utility
