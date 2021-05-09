import random
import sys
from environment import Environment
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
    POSX = (1, 0, 0)
    NEGX = (-1, 0, 0)
    NEGY = (0, -1, 0)
    POSY = (0, 1, 0)
    POSZ = (0, 0, 1)
    NEGZ = (0, 0, -1)

    MOVES = [NEGY, POSY, POSX, NEGX, POSZ, NEGZ]

    def __init__(self, x, y, z, space_width, space_height, space_depth, epsilon_decay=0.95, epsilon=1):
        self.x = x
        self.y = y
        self.z = z
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

        self.utility_table = np.zeros((self.width, self.height, self.depth))

    def distance_to(self, point):
        manhattan_distance = abs(self.x - point[0]) + abs(self.y - point[1]) + abs(self.z - point[2])
        return manhattan_distance

    def location_string(self):
        return '(%s, %s, %s)' % (self.x, self.y, self.z)

    def last_pollution(self):
        if len(self.experiences) == 0:
            return 0
        return self.experiences[len(self.experiences) - 1][1]

    def last_do2(self):
        if len(self.experiences) == 0:
            return 0
        return self.experiences[len(self.experiences) - 1][2]

    def potential_moves(self):
        moves = []
        if self.x > 0:
            moves.append(Agent.NEGX)

        if self.x < self.width - 1:
            moves.append(Agent.POSX)

        if self.y > 0:
            moves.append(Agent.NEGY)

        if self.y < self.height - 1:
            moves.append(Agent.POSY)

        if self.z > 0:
            moves.append(Agent.NEGZ)

        if self.z < self.depth - 1:
            moves.append(Agent.POSZ)

        return moves

    def choose_action(self, environment: Environment):

        # perceive environment status based on location
        pollution = environment.get_pollution(self.x, self.y, self.z)
        do2 = environment.get_do2(self.x, self.y, self.z)

        # if agent is at source it will do nothing
        if pollution == Environment.SOURCE:
            debug_print('source found at, %s. Pollution: %s do2: %s' % (self.location_string(), pollution, do2))
            print('Found Source')
            return Agent.NONE

        # Error checking
        if self.x < 0 or self.y < 0 or self.z < 0:
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
                        % (action, self.location_string(), pollution, do2))
            return action


        # Exploitation of Action Policy
        # exploit
        best_utility = None
        best_move = None
        for move in moves:
            debug_print('Checking move %s.' % (move, ))
            x, y, z = self.next_local(move)
            utility = self.utility_table[x, y, z]
            if best_utility is None or utility > best_utility:
                best_utility = utility
                best_move = move

        debug_print('choose based on utility')
        return best_move

        # if pollution is increasing and DO2 is decreasing the agent will return the previous action otherwise random.
        # if do2 < self.last_do2():
        #     debug_print('Moving %s, from %s pollution: %s do2: %s.'
        #                 % (self.previous_action, self.location_string(), pollution, do2))
        #     return self.previous_action
        #
        # if do2 == self.last_do2():
        #     if pollution >= self.last_pollution():
        #         debug_print('Moving %s, from %s pollution: %s do2: %s.'
        #                     % (self.previous_action, self.location_string(), pollution, do2))
        #         return self.previous_action
        #     else:
        #         action = random.choice((Agent.NEGX, Agent.POSX, Agent.NEGY, Agent.POSY, Agent.POSZ, Agent.NEGZ))
        #         debug_print('Moving %s from %s pollution: %s , DO2: %s'
        #                     % (action, self.location_string(), pollution, do2))
        #         return action
        #
        # if do2 > self.last_do2():
        #     action = random.choice((Agent.NEGX, Agent.POSX, Agent.NEGY, Agent.POSY, Agent.POSZ, Agent.NEGZ))
        #     # debug_print('DO2 increasing. Moving %s' % action)
        #     return action
        #
        # print("unknown status:", pollution)

    def is_at(self, x, y, z):
        # print(x, y, z)
        return self.x == x and self.y == y and self.z == z

    def update_utilities(self):
        exp_count = len(self.experiences)
        trial_u = [0] * exp_count
        for exp_idx in range(exp_count, 0, -1):
            experience = self.experiences[exp_idx-1]
            exp_reward = experience[3]
            for r_idx in range(exp_idx, 0, -1):
                trial_u[r_idx-1] += exp_reward
                exp_reward *= self.gamma

        # alpha is 0.5 (utility+trialu/2). adjust alpha by changing denominator
        for exp_idx in range(exp_count):
            experience = self.experiences[exp_idx]
            x, y, z = experience[0]
            current_utility = self.utility_table[x, y, z]
            self.utility_table[x, y, z] = (current_utility + trial_u[exp_idx]) / 2

    def print_utilities(self):
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    print('{:.2f} '.format(self.utility_table[x, y, z]), end='')
            print()

    def reset(self):
        # todo once total steps are normalized in the project.py reset to random positions
        self.epsilon *= self.epsilon_decay
        self.x = random.randint(0, self.width-1)
        self.y = random.randint(0, self.height-1)
        self.z = random.randint(0, self.depth-1)
        self.experiences.clear()

    # def reward(self, environment):
    #     reward = 0
    #     if environment.is_goal(self.x, self.y):
    #         reward = REWARD
    #     return reward

    # Pollution reward
    def reward(self, environment):
        if environment.get_pollution(self.x, self.y, self.z) == Environment.SOURCE:
            return 100
        return 0

    def act(self, action, environment: Environment):
        # allows agent to know action and status of prior environment
        pollution = environment.get_pollution(self.x, self.y, self.z)
        do2 = environment.get_do2(self.x, self.y, self.z)

        self.previous_action = action

        # if action = NONE that means agent is at the source and should stop.
        if action == Agent.NONE:
            return

        # in case the action is left location equals location - 1
        if action == Agent.NEGX:
            self.move_NEGX()

        #   in case the action is right location equals location + 1
        if action == Agent.POSX:
            self.move_POSX()

        #   in case the action is right location equals location + 1
        if action == Agent.POSY:
            self.move_POSY()

        #   in case the action is right location equals location + 1
        if action == Agent.NEGY:
            self.move_NEGY()

        #   in case the action is right location equals location + 1
        if action == Agent.POSZ:
            self.move_POSZ()

        #   in case the action is left location equals location - 1
        if action == Agent.NEGZ:
            self.move_NEGZ()

        reward = self.reward(environment)
        # Make experiences a dict with key being x, y, z
        self.experiences.append(
            (
                (self.x, self.y, self.z),
                pollution,
                do2,
                reward
            )
        )

    def move(self, delta_x, delta_y, delta_z):
        self.x += delta_x
        self.y += delta_y
        self.z += delta_z

    def move_POSX(self):
        self.move(1, 0, 0)

    def move_NEGX(self):
        self.move(-1, 0, 0)

    def move_NEGY(self):
        self.move(0, -1, 0)

    def move_POSY(self):
        self.move(0, 1, 0)

    def move_POSZ(self):
        self.move(0, 0, 1)

    def move_NEGZ(self):
        self.move(0, 0, -1)

    def location(self):
        return self.x, self.y, self.z

    def set_utility(self, location, utility):
        self.utility_table[location] = utility

    def next_local(self, move):
        return self.x + move[0], self.y + move[1], self.z + move[2]
