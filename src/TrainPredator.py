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
from Statistics_task3 import StatisticsTask3
from tqdm import tqdm, trange
import socket
import cv2
import prey_controller as prey
import pandas as pd
from itertools import product


MULTIPLE_RUNS = False  # Doing an experiment multiple times, not required for normal training.
N_RUNS = 5  # How many times an experiment is done if MULTIPLE_RUNS = True.
EXPERIMENT_COUNTER = 0  # Only needed for training over multiple experiments (MULTIPLE_RUNS = "True")

# For each time training, give this a unique name so the data can be saved with a unique name.
EXPERIMENT_NAME = 'test'


class Direction:
    LEFT = (-5, 5, 300)  # Action: 0, left
    RIGHT = (5, -5, 300)  # Action: 1, right
    FORWARD = (25, 25, 300)  # Action: 2, forward
    RRIGHT = (-15, 15, 300)  # Action: 3, strong right
    LLEFT = (15, -15, 300)  # Action: 4, strong left


# noinspection PyProtectedMember
class Environment:
    # All of our constants that together define a training set-up.
    MAX_ITERATIONS = 50  # Amount of simulations until termination.
    MAX_SIMULATION_ITERATIONS = 250  # Amount of actions within one simulation. Actions = Q-table updates.

    LEARNING_RATE = .1
    DISCOUNT_FACTOR = .9
    EPSILON = 0.9  # Start epsilon value.

    IP_ADDRESS = socket.gethostbyname(socket.gethostname())  # Grabs local IP address (192.168.x.x) for your machine.

    action_space = [0, 1, 2, 3, 4]  # All of our available actions. Find definitions in the Direction class.
    action_space_prey = [0, 1, 2]
    iteration_counter, physical_collision_counter = 0, 0

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)
        self.rob.play_simulation()
        self.prey = robobo.SimulationRoboboPrey().connect(address=self.IP_ADDRESS, port=19989)

        q_table_prey = self.initialize_q_table_prey()
        self.prey_controller = prey.Prey(robot=self.prey, q_table=q_table_prey, level=2)
        self.prey_controller.start()

        # Stuff for keeping track of stats/data
        self.stats = StatisticsTask3(self.MAX_ITERATIONS, self.MAX_SIMULATION_ITERATIONS)
        self.q_table = self.initialize_q_table()

        _, self.collision_handle = vrep.simxGetCollisionHandle(self.rob._clientID, 'Collision',
                                                               vrep.simx_opmode_blocking)

    def start_environment(self):
        for i in trange(self.MAX_ITERATIONS):  # Nifty, innit?
            self.rob.set_phone_tilt(np.pi / 6, 100)

            total_prey_reward = 0
            total_predator_reward = 0

            # A simulation runs until valid_environment returns False.
            while self.valid_environment():
                # Check in what state rob is, return tuple e.g. (0, 0, 0, 1, 0). A state is defined by rob's sensors.
                curr_state = self.handle_state()

                # Do we perform random action (due to epsilon < 1) or our best possible action?
                best_action = self.determine_action(curr_state)

                # Given our selected action (whether best or random), perform this action and update the Q-table.
                total_predator_reward += self.update_q_table(i, best_action, curr_state)

                total_prey_reward += self.prey_controller.get_reward()

                # self.stats.add_reward(i, self.iteration_counter, reward)  # Add the reward for visualization purposes.
                self.iteration_counter += 1  # Keep track of how many actions this simulation does.

            # self.stats.add_step_counter(i, self.iteration_counter)
            if self.physical_collision_counter > 0:
                self.stats.catched(i, self.iteration_counter)

            self.stats.add_rewards(i, total_predator_reward, total_prey_reward)

            self.iteration_counter = 0
            self.physical_collision_counter = 0

            q_prey = self.prey_controller.q_table
            self.prey_controller.stop()
            self.prey_controller.join()
            self.prey.disconnect()
            self.rob.stop_world()
            self.rob.wait_for_ping()
            self.rob.play_simulation()
            self.prey = robobo.SimulationRoboboPrey().connect(address=self.IP_ADDRESS, port=19989)
            self.prey_controller = prey.Prey(robot=self.prey, q_table=q_prey, level=2)
            self.prey_controller.start()
        self.stats.save_data(EXPERIMENT_NAME)

    def best_action_for_state(self, state):
        # Given a state (tuple format), what is the best action we take, i.e. for which action is the Q-value highest?
        q_row = self.q_table[state]
        max_val_indices = [i for i, j in enumerate(q_row) if j == max(q_row)]
        best_action = random.choice(max_val_indices) if len(max_val_indices) > 1 else np.argmax(q_row)

        return best_action

    def physical_collision(self):
        # This function checks for physical collision that is not based on the sensors, but on an object around the Robot
        # in V-REP.
        [_, collision_state] = vrep.simxReadCollision(self.rob._clientID, self.collision_handle, vrep.simx_opmode_streaming)
        return collision_state

    def determine_action(self, curr_state):
        if random.random() < (1 - self.EPSILON):
            return random.choice(self.action_space)
        else:
            return self.best_action_for_state(curr_state)

    @staticmethod
    def terminate_program(test1, test2):
        # Only do this for training and not for testing, to avoid overwriting a valid Q-table.
        print(f"Ctrl-C received, terminating program.")
        sys.exit(1)

    def valid_environment(self):
        # This function checks whether the current simulation can continue or not, depending on several criteria.
        c1 = self.iteration_counter >= self.MAX_SIMULATION_ITERATIONS
        c2 = self.physical_collision_counter > 0

        return False if any([c1, c2]) else True

    def initialize_q_table(self):
        # Initialize Q-table for states * action pairs with default values (0).
        # E.g. the size (5, 5, 5, 5, 5) denotes each sensor, with its amount of possible states (see func handle_state).
        return np.random.uniform(low=6, high=6, size=([2, 2, 2] + [len(self.action_space)]))

    def initialize_q_table_prey(self):
        # Initialize Q-table for states * action pairs with default values (0).
        # E.g. the size (5, 5, 5, 5, 5) denotes each sensor, with its amount of possible states (see func handle_state).
        return np.random.uniform(low=6, high=6, size=([3, 3, 3, 3, 3, 3, 3, 3] + [len(self.action_space_prey)]))

    def handle_state(self):
        contours_left, contours_center, contours_right = self.determine_food()
        res = tuple()

        if contours_left > 0:
            res += (1,)
        else:
            res += (0,)

        if contours_center > 0:
            res += (1,)
        else:
            res += (0,)

        if contours_right > 0:
            res += (1,)
        else:
            res += (0,)

        # print(f'State: {res}')
        return res

    def determine_food(self):
        image = self.rob.get_image_front()

        maskl = np.zeros(image.shape, dtype=np.uint8)
        maskr = np.zeros(image.shape, dtype=np.uint8)
        maskc = np.full(image.shape, 255, dtype=np.uint8)

        maskc = cv2.ellipse(maskc, (0, 128), (50, 90), 180, 0, 180, (0, 0, 0), -1)
        maskc = cv2.ellipse(maskc, (128, 128), (50, 90), 180, 0, 180, (0, 0, 0), -1)

        maskl = cv2.ellipse(maskl, (0, 128), (50, 90), 180, 0, 180, (255, 255, 255), -1)
        maskr = cv2.ellipse(maskr, (128, 128), (50, 90), 180, 0, 180, (255, 255, 255), -1)

        resultc = cv2.bitwise_and(image, maskc)
        resultl = cv2.bitwise_and(image, maskl)
        resultr = cv2.bitwise_and(image, maskr)

        mask_left = cv2.inRange(resultl, (0, 0, 100), (140, 140, 255))
        mask_center = cv2.inRange(resultc, (0, 0, 100), (140, 140, 255))
        mask_right = cv2.inRange(resultr, (0, 0, 100), (140, 140, 255))

        contours_left, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_center, _ = cv2.findContours(mask_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return len(contours_left), len(contours_center), len(contours_right)

    def handle_action(self, i, action, curr_state):
        # This function should accept an action (0, 1, 2...) and move the robot accordingly (left, right, forward).
        # It returns two things: new_state, which is the state (in tuple format) after this action has been performed.
        # and reward, which is the reward from this action.

        collision2 = self.physical_collision()

        if collision2:
            self.physical_collision_counter += 1

        if action == 0:
            left, right, duration = Direction.LEFT
        elif action == 1:
            left, right, duration = Direction.RIGHT
        elif action == 2:
            left, right, duration = Direction.FORWARD
        elif action == 3:
            left, right, duration = Direction.LLEFT
        else:
            left, right, duration = Direction.RRIGHT

        if i < 10:
            left, right, duration = 0, 0, 300

        self.rob.move(left, right, duration)

        reward = self.determine_reward(action, curr_state)
        return self.handle_state(), reward  # New_state, reward

    def determine_reward(self, action, curr_state):
        # This function determines the reward an action should get
        # Actions: left, right, forward, rright, lleft
        reward = 0
        if self.physical_collision_counter > 0:
            reward += 100

        if curr_state[1] > 0 and action == 2:
            reward += 5

        if curr_state[0] == 0 and curr_state[1] == 0 and curr_state[2] == 0:  # We see nothing, turn
            if action in [0, 1, 2, 3]:  # If we do anything except for a hard turn, punish
                reward -= 2
            elif action in [4]:  # We reward a hard turn if we see nothing
                reward += 5

        return reward

    def update_q_table(self, i, best_action, curr_state):
        # This function updates the Q-table accordingly to the current state of rob.
        # First, we determine the new state we end in if we would play our current best action, given our current state.
        new_state, reward = self.handle_action(i, best_action, curr_state)

        # Then we calculate the reward we would get in this new state.
        max_future_q = np.amax(self.q_table[new_state])

        # Check what Q-value our current action has.
        current_q = self.q_table[curr_state][best_action]

        # Calculate the new Q-value with the common formula
        new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (
                reward + self.DISCOUNT_FACTOR * max_future_q)

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
            print(f"Begin experiment {i + 1}/{N_RUNS}")
            env.EPSILON_LOW = epsilon_low
            EXPERIMENT_COUNTER += 1

            exp_name = f"{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}_{EXPERIMENT_COUNTER}.pickle"
            filename_rewards = f"results/{EXPERIMENT_NAME}/reward_data_" + exp_name
            filename_q_table = f"results/{EXPERIMENT_NAME}/q_table_data_" + exp_name
            env.q_table = env.initialize_q_table()
            env.start_environment()
            # env.stats.save_rewards(filename_rewards)
            env.store_q_table(filename_q_table)

    else:
        env.start_environment()

        # Save all data (rewards, food collected, steps done per simulation, and the Q-table.
        # env.stats.save_rewards(
        #     f"results/reward_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle")
        # env.store_q_table(
        #     f"results/q_table_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle")


if __name__ == "__main__":
    main()