#!/usr/bin/env python3
from __future__ import print_function
import time
import numpy as np
import robobo
import sys
import signal
import prey
import pickle
import random
import os
import matplotlib.pyplot as plt
import pprint
from Statistics import Statistics
from datetime import datetime
import seaborn as sns
from tqdm import tqdm, trange
import socket


MULTIPLE_RUNS = False  # Doing an experiment multiple times, not required for normal training.
N_RUNS = 5  # How many times an experiment is done if MULTIPLE_RUNS = True.
EXPERIMENT_COUNTER = 0  # Only needed for training over multiple experiments (MULTIPLE_RUNS = "True")

# For each time training, give this a unique name so the data can be saved with a unique name.
EXPERIMENT_NAME = 'test1'


class Direction:
    LEFT = (-15, 15, 300)  # Action: 0, left
    RIGHT = (15, -15, 300)  # Action: 1, right
    FORWARD = (25, 25, 300)  # Action: 2, forward
    RRIGHT = (25, -25, 300)  # Action: 3, strong right
    LLEFT = (-25, 25, 300)  # Action: 4, strong left


class Environment:
    # All of our constants that together define a training set-up.
    MAX_ITERATIONS = 20  # Amount of simulations until termination.
    MAX_SIMULATION_ITERATIONS = 200  # Amount of actions within one simulation. Actions = Q-table updates.

    LEARNING_RATE = .1
    DISCOUNT_FACTOR = .95
    EPSILON_LOW = 0.6  # Start epsilon value. This gradually increases.
    EPSILON_HIGH = 1  # End epsilon value
    EPSILON_INCREASE = .01  # How much should we increase the epsilon value with, each time?

    IP_ADDRESS = socket.gethostbyname(socket.gethostname())  # Grabs local IP address (192.168.x.x) for your machine.

    COLLISION_THRESHOLD = 100  # After how many collision actions should we reset the environment? Prevents rob getting stuck.

    action_space = [0, 1, 2, 3, 4]  # All of our available actions. Find definitions in the Direction class.
    collision_counter, iteration_counter, epsilon_counter = 0, 0, 0

    # The epsilon_increase determines when the epsilon should be increased. This happens gradually from EPSILON_LOW
    # to EPSILON_HIGH during the amount of allowed iterations. So when MAX_ITERATIONS reaches its limit, so does
    # the epsilon value.
    epsilon_increase = int(((MAX_ITERATIONS * MAX_SIMULATION_ITERATIONS) // (EPSILON_HIGH - EPSILON_LOW) * 100) / 10_000)

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)
        self.stats = Statistics(self.MAX_ITERATIONS, self.MAX_SIMULATION_ITERATIONS)
        self.q_table = self.initialize_q_table()

    def start_environment(self):
        for i in trange(self.MAX_ITERATIONS):  # Nifty, innit?
            print(f"Starting simulation nr. {i+1}/{self.MAX_ITERATIONS}. Epsilon: {self.EPSILON_LOW}. Q-table size: {self.q_table.size}")

            self.rob.wait_for_ping()
            self.rob.play_simulation()

            # A simulation runs until valid_environment returns False.
            while self.valid_environment():
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
                self.store_q_table()  # Save Q-table after each iteration because, why not.

                # Reset the counters
                self.iteration_counter = 0
                self.collision_counter = 0

                self.rob.stop_world()
                self.rob.wait_for_ping()  # Maybe we should wait for ping so we avoid errors. Might not be necessary.

    @staticmethod
    def read_q_table(filename):
        # This function loads an existing Q-table.
        with open(filename, 'rb') as fp:
            q_table = pickle.load(fp)
        return q_table

    def store_q_table(self):
        with open(f"results/q_table_{self.MAX_ITERATIONS}_{self.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle", 'wb') as fp:
            pickle.dump(self.q_table, fp)

    def best_action_for_state(self, state):
        # Given a state (tuple format), what is the best action we take, i.e. for which action is the Q-value highest?
        q_row = self.q_table[state]
        max_val_indices = [i for i, j in enumerate(q_row) if j == max(q_row)]
        best_action = random.choice(max_val_indices) if len(max_val_indices) > 1 else np.argmax(q_row)

        return best_action

    def determine_action(self, curr_state):
        return random.choice(self.action_space) if random.random() < (1 - self.EPSILON_LOW) else self.best_action_for_state(curr_state)

    @staticmethod
    def terminate_program(self, test1, test2):
        # Only do this for training and not for testing, to avoid overwriting a valid Q-table.
        print(f"Ctrl-C received, terminating program.")
        self.store_q_table()
        sys.exit(1)

    def valid_environment(self):
        # This function checks whether the current simulation can continue or not, depending on several criteria.
        c1 = self.collision_counter > self.COLLISION_THRESHOLD
        c2 = self.iteration_counter >= self.MAX_SIMULATION_ITERATIONS

        return False if any([c1, c2]) else True

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
        return np.random.uniform(low=-.1, high=.1, size=([3, 3, 3, 3, 3] + [len(self.action_space)]))

    def handle_state(self):
        # This function should return the values with which we can index our q_table, in tuple format.
        # So, it should take the last 5 sensor inputs (current state), transform each of them into a bucket where
        # the bucket size is already determined by the shape of the q_table.
        try:
            sensor_values = np.log(np.array(self.rob.read_irs()[3:])) / 10  #
        except:
            sensor_values = [0, 0, 0, 0, 0]

        sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
        sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

        # Check what the actual sensor_values are (between [0, 1]) and determine their state
        indices = []
        for sensor_value in sensor_values:
            if sensor_value >= 0.8:  # No need for action, moving forward is best.
                indices.append(0)
            elif 0.65 <= sensor_value < 0.8:
                indices.append(1)
            elif sensor_value < 0.65:  # We see an object, but not really close yet.
                indices.append(2)

        # Return the values in tuple format, with which we can index our Q-table. This tuple is a representation
        # of the current state our robot is in (i.e. what does the robot see with its sensors).
        return tuple(indices)

    def handle_action(self, action):
        # This function should accept an action (0, 1, 2...) and move the robot accordingly (left, right, forward).
        # It returns two things: new_state, which is the state (in tuple format) after this action has been performed.
        # and reward, which is the reward from this action.
        collision = self.collision()  # Do we collide, returns either "nothing", "far" or "close"
        reward = self.determine_reward(collision, action)

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
    def determine_reward(collision, action):
        # This function determines the reward an action should get, depending on whether or not rob is about to
        # collide with an object within the environment.
        reward = 0

        if action in [0, 1, 3, 4]:  # Action is moving either left or right.
            if collision == "nothing":
                reward -= 1
            elif collision == "far":
                reward += 1
            elif collision == "close":
                reward += 2
        elif action == 2:  # Action is moving forward.
            if collision == "far":
                reward -= 3
            elif collision == "close":
                reward -= 5
            elif collision == "nothing":
                reward += 3

        return reward

    def collision(self):
        # This function checks whether rob is close to something or not. It returns the "distance", either "close", "far" or "nothing".
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        try:
            sensor_values = self.rob.read_irs()[3:]  # Should be absolute values (no log or anything).
        except:
            sensor_values = [0, 0, 0, 0, 0]

        collision_far = any([0.13 <= i < 0.2 for i in sensor_values])
        collision_close = any([0 < i < 0.13 for i in sensor_values])

        if collision_close:
            self.collision_counter += 1
            return "close"
        elif collision_far:
            return "far"
        else:
            self.collision_counter = 0
            return "nothing"

    def update_q_table(self, best_action, curr_state):
        # This function updates the Q-table accordingly to the current state of rob.
        # First, we determine the new state we end in if we would play our current best action, given our current state.
        new_state, reward = self.handle_action(best_action)

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
            file_name = f"results/{EXPERIMENT_NAME}/reward_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}_{EXPERIMENT_COUNTER}.pickle"
            env.q_table = env.initialize_q_table()
            env.start_environment()
            env.stats.save_rewards(file_name)
    else:
        env.start_environment()
        env.stats.save_rewards(f"results/reward_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle")


if __name__ == "__main__":
    main()
