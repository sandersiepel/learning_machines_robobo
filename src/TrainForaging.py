#!/usr/bin/env python3
from __future__ import print_function
import time
import numpy as np
import robobo
import sys
import vrep
import signal
import prey
import pickle
import random
import os
from Statistics import Statistics
from tqdm import tqdm, trange
import socket
import cv2


MULTIPLE_RUNS = False  # Doing an experiment multiple times, not required for normal training.
N_RUNS = 5  # How many times an experiment is done if MULTIPLE_RUNS = True.
EXPERIMENT_COUNTER = 0  # Only needed for training over multiple experiments (MULTIPLE_RUNS = "True")

# For each time training, give this a unique name so the data can be saved with a unique name.
EXPERIMENT_NAME = 'train_week1_1'


class Direction:
    LEFT = (-2, 2, 300)  # Action: 0, left
    RIGHT = (2, -2, 300)  # Action: 1, right
    FORWARD = (25, 25, 300)  # Action: 2, forward
    RRIGHT = (15, -15, 300)  # Action: 3, strong right
    LLEFT = (-15, 15, 300)  # Action: 4, strong left


# noinspection PyProtectedMember
class Environment:
    # All of our constants that together define a training set-up.
    MAX_ITERATIONS = 10  # Amount of simulations until termination.
    MAX_SIMULATION_ITERATIONS = 200  # Amount of actions within one simulation. Actions = Q-table updates.
    FOOD_AMOUNT = 6

    LEARNING_RATE = .1
    DISCOUNT_FACTOR = .95
    EPSILON_LOW = 0.9  # Start epsilon value. This gradually increases.
    EPSILON_HIGH = 0.91  # End epsilon value
    EPSILON_INCREASE = 0  # How much should we increase the epsilon value with, each time?

    IP_ADDRESS = socket.gethostbyname(socket.gethostname())  # Grabs local IP address (192.168.x.x) for your machine.

    action_space = [0, 1, 2, 3, 4]  # All of our available actions. Find definitions in the Direction class.
    iteration_counter, epsilon_counter = 0, 0

    # The epsilon_increase determines when the epsilon should be increased. This happens gradually from EPSILON_LOW
    # to EPSILON_HIGH during the amount of allowed iterations. So when MAX_ITERATIONS reaches its limit, so does
    # the epsilon value.
    epsilon_increase = int((((MAX_ITERATIONS * MAX_SIMULATION_ITERATIONS) // (EPSILON_HIGH - EPSILON_LOW) * 100) / 10_000) * 0.7)

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)
        self.stats = Statistics(self.MAX_ITERATIONS, self.MAX_SIMULATION_ITERATIONS)
        self.q_table = self.initialize_q_table()

    def start_environment(self):
        # Initialize the start position handles + the robot handle
        robot, handles = self.initialize_handles()

        for i in trange(self.MAX_ITERATIONS):  # Nifty, innit?
            # print(f"Starting simulation nr. {i+1}/{self.MAX_ITERATIONS}. Epsilon: {self.EPSILON_LOW}. Q-table size: {self.q_table.size}")

            # Each new simulation, find a new (random) start position based on the provided handles.
            # start_pos, robot = self.determine_start_position(handles, robot)
            # vrep.simxSetObjectPosition(self.rob._clientID, robot, -1, start_pos, vrep.simx_opmode_oneshot)
            self.rob.play_simulation()

            # A simulation runs until valid_environment returns False.
            while self.valid_environment():
                print(self.EPSILON_LOW)
                # Check in what state rob is, return tuple e.g. (0, 0, 0, 1, 0). A state is defined by rob's sensors.
                curr_state = self.handle_state()

                # Do we perform random action (due to epsilon < 1) or our best possible action?
                best_action = self.determine_action(curr_state)

                # Given our selected action (whether best or random), perform this action and update the Q-table.
                reward = self.update_q_table(best_action, curr_state)

                self.stats.add_reward(i, self.iteration_counter, reward)  # Add the reward for visualization purposes.
                self.change_epsilon()  # Check if we should increase epsilon or not.
                self.iteration_counter += 1  # Keep track of how many actions this simulation does.
            else:
                print(self.q_table)
                # self.store_q_table()  # Save Q-table after each iteration because, why not.
                # Reset the counters
                self.iteration_counter = 0
                self.rob.stop_world()
                self.rob.wait_for_ping()  # Maybe we should wait for ping so we avoid errors. Might not be necessary.

            print(self.q_table)

    @staticmethod
    def read_q_table(filename):
        # This function loads an existing Q-table.
        with open(filename, 'rb') as fp:
            q_table = pickle.load(fp)
        return q_table

    def store_q_table(self, name):
        with open(name, 'wb') as fp:
            pickle.dump(self.q_table, fp)

    def initialize_handles(self):
        # This function initializes the starting handles. These are used for initializing different start positions.
        _, robot = vrep.simxGetObjectHandle(self.rob._clientID, 'Robobo', vrep.simx_opmode_blocking)
        _, handle0 = vrep.simxGetObjectHandle(self.rob._clientID, 'Start#0', vrep.simx_opmode_blocking)
        _, handle1 = vrep.simxGetObjectHandle(self.rob._clientID, 'Start#1', vrep.simx_opmode_blocking)
        _, handle2 = vrep.simxGetObjectHandle(self.rob._clientID, 'Start#2', vrep.simx_opmode_blocking)
        _, handle2 = vrep.simxGetObjectHandle(self.rob._clientID, 'Start#3', vrep.simx_opmode_blocking)

        return robot, [handle0, handle1, handle2]

    def determine_start_position(self, handles, robot):
        # Given a handle for a start position, and a robot handle: determine a new start location randomly.
        random_handle = random.choice(handles)
        error, start_pos = vrep.simxGetObjectPosition(self.rob._clientID, random_handle, -1, vrep.simx_opmode_blocking)
        return [start_pos[0], start_pos[1], 0.04], robot

    def best_action_for_state(self, state):
        # Given a state (tuple format), what is the best action we take, i.e. for which action is the Q-value highest?
        q_row = self.q_table[state]
        max_val_indices = [i for i, j in enumerate(q_row) if j == max(q_row)]
        best_action = random.choice(max_val_indices) if len(max_val_indices) > 1 else np.argmax(q_row)

        return best_action

    def determine_action(self, curr_state):
        if random.random() < (1 - self.EPSILON_LOW):
            print("random choice")
            return random.choice(self.action_space)
        else:
            return self.best_action_for_state(curr_state)

    @staticmethod
    def terminate_program(self, test1, test2):
        # Only do this for training and not for testing, to avoid overwriting a valid Q-table.
        print(f"Ctrl-C received, terminating program.")
        self.store_q_table()
        sys.exit(1)

    def valid_environment(self):
        # This function checks whether the current simulation can continue or not, depending on several criteria.
        c2 = self.iteration_counter >= self.MAX_SIMULATION_ITERATIONS
        c3 = self.rob.collected_food() >= self.FOOD_AMOUNT

        return False if any([c2, c3]) else True

    def change_epsilon(self):
        # This function changes the epsilon value if needed. Only does so if we did x amount of iterations, and the
        # current epsilon value is smaller than the epsilon limit (EPSILON_HIGH).
        if self.epsilon_counter == self.epsilon_increase:
            if self.EPSILON_LOW < self.EPSILON_HIGH:
                self.EPSILON_LOW += self.EPSILON_INCREASE
                self.epsilon_counter = 0
        else:
            self.epsilon_counter += 1

    def initialize_q_table(self):
        # Initialize Q-table for states * action pairs with default values (0).
        # Since observation space is very large, we need to trim it down (bucketing) to only a select amount of
        # possible states, e.g. 4 for each sensor (4^8 = 65k). Or: use less sensors (no rear sensors for task 1).
        # E.g. the size (5, 5, 5, 5, 5) denotes each sensor, with its amount of possible states (see func handle_state).
        return np.random.uniform(low=6, high=6, size=([2, 2] + [len(self.action_space)]))

    def handle_state(self):
        contours_left, contours_right = self.determine_food()

        res = tuple()
        if contours_left > 0:
            res += (1,)
        else:
            res += (0,)
        if contours_right > 0:
            res += (1,)
        else:
            res += (0,)

        return res

    def determine_food(self):
        image = self.rob.get_image_front()

        # Chop image vertically in two
        image_left = image[:, 0:64, :]
        image_right = image[:, 64:128, :]

        # Create mask for color green
        mask_left = cv2.inRange(image_left, (0, 100, 0), (90, 255, 90))
        mask_right = cv2.inRange(image_right, (0, 100, 0), (90, 255, 90))

        # Find contours, if present
        contours_left, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return len(contours_left), len(contours_right)

    def handle_action(self, action, curr_state):
        # This function should accept an action (0, 1, 2...) and move the robot accordingly (left, right, forward).
        # It returns two things: new_state, which is the state (in tuple format) after this action has been performed.
        # and reward, which is the reward from this action.

        reward = self.determine_reward(action, curr_state)

        if action == 0:
            left, right, duration = Direction.LEFT  # Left, action 0
        elif action == 1:
            left, right, duration = Direction.RIGHT  # Right, action 1
        elif action == 3:
            left, right, duration = Direction.RRIGHT  # Extreme right, action 3
        elif action == 4:
            left, right, duration = Direction.LLEFT  # Extreme left, action 4
        else:
            left, right, duration = Direction.FORWARD  # Forward, action 2

        self.rob.move(left, right, duration)
        return self.handle_state(), reward  # New_state, reward

    @staticmethod
    def determine_reward(action, curr_state):
        # This function determines the reward an action should get, depending on whether or not rob is about to
        # collide with an object within the environment.\
        # Actions: left, right, forward, lleft, rright
        reward = 0

        if curr_state[0] == 0 and curr_state[1] == 0:
            # Rob sees no food, we should explore (turn)
            print('We see no food')
            if action == 3:  # Only promote turning left to avoid left-right-left-right behavior
                reward += 5
            else:
                reward -= 5

        if curr_state[0] > 0 and curr_state[1] == 0:
            # Only something on our left, we should move left a bit
            print("Food on left")
            if action == 0:
                reward += 5
            else:
                reward -= 5

        if curr_state[0] == 0 and curr_state[1] > 0:
            # Only something on our right, we should move right a bit
            print("Food on right")
            if action == 1:
                reward += 5
            else:
                reward -= 5

        if curr_state[0] > 0 and curr_state[1] > 0:
            # Something in the middle OR both left and right has objects, move forward
            print("Food on left AND right")
            if action == 2:
                reward += 5
            else:
                reward -= 5

        print(f"Curr state: {curr_state}, action: {action}, received reward: {reward}\n")
        return reward

    def update_q_table(self, best_action, curr_state):
        # This function updates the Q-table accordingly to the current state of rob.
        # First, we determine the new state we end in if we would play our current best action, given our current state.
        new_state, reward = self.handle_action(best_action, curr_state)

        # Then we calculate the reward we would get in this new state.
        max_future_q = np.amax(self.q_table[new_state])

        # Check what Q-value our current action has.
        current_q = self.q_table[curr_state][best_action]

        # Calculate the new Q-value with the common formula
        new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * max_future_q)

        # And lastly, update the value in the Q-table.
        self.q_table[curr_state][best_action] = new_q

        return reward


def main():
    env = Environment()

    if MULTIPLE_RUNS:  # Option to do multiple runs
        # Check if rewards folder exists, if not: create it.
        if not os.path.exists(f'results/{EXPERIMENT_NAME}'):
            os.makedirs(f'results/{EXPERIMENT_NAME}')

        epsilon_low = env.EPSILON_LOW
        global EXPERIMENT_COUNTER

        for i in range(N_RUNS):
            print(f"Begin experiment {i+1}/{N_RUNS}")
            env.EPSILON_LOW = epsilon_low
            EXPERIMENT_COUNTER += 1

            # TODO (not necessary); also store q-table per experiment
            exp_name = f"{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}_{EXPERIMENT_COUNTER}.pickle"
            filename_rewards = f"results/{EXPERIMENT_NAME}/reward_data_" + exp_name
            filename_collision = f"results/{EXPERIMENT_NAME}/collision_data_" + exp_name
            filename_q_table = f"results/{EXPERIMENT_NAME}/q_table_data_" + exp_name
            env.q_table = env.initialize_q_table()
            env.start_environment()
            env.stats.save_rewards(filename_rewards)
            env.stats.save_collision(filename_collision)
            env.store_q_table(filename_q_table)

    else:
        env.start_environment()
        # filename_rewards = f"results/reward_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle"
        # filename_collision = f"results/collision_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle"
        # filename_q_table = f"results/q_table_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle"
        # env.stats.save_rewards(filename_rewards)
        # env.stats.save_collision(filename_collision)
        # env.store_q_table(filename_q_table)


if __name__ == "__main__":
    main()
