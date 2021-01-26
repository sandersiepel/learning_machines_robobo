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
import prey_controller as prey
import pandas as pd
from itertools import product


MULTIPLE_RUNS = False  # Doing an experiment multiple times, not required for normal training.
N_RUNS = 5  # How many times an experiment is done if MULTIPLE_RUNS = True.
EXPERIMENT_COUNTER = 0  # Only needed for training over multiple experiments (MULTIPLE_RUNS = "True")

# For each time training, give this a unique name so the data can be saved with a unique name.
EXPERIMENT_NAME = 'predator'


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
    MAX_SIMULATION_ITERATIONS = 200  # Amount of actions within one simulation. Actions = Q-table updates.

    LEARNING_RATE = .1
    DISCOUNT_FACTOR = .8
    EPSILON = 0.9  # Start epsilon value.

    IP_ADDRESS = socket.gethostbyname(socket.gethostname())  # Grabs local IP address (192.168.x.x) for your machine.

    action_space = [0, 1, 2, 3, 4]  # All of our available actions. Find definitions in the Direction class.
    iteration_counter = 0

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)
        self.rob.play_simulation()
        self.prey = robobo.SimulationRoboboPrey().connect(address=self.IP_ADDRESS, port=19989)

        q_table_prey = self.initialize_q_table_prey()
        self.prey_controller = prey.Prey(robot=self.prey, q_table=q_table_prey, level=2)
        self.prey_controller.start()

        # Stuff for keeping track of stats/data
        self.stats = Statistics(self.MAX_ITERATIONS, self.MAX_SIMULATION_ITERATIONS)
        self.q_table = self.initialize_q_table()

    def start_environment(self):
        for i in trange(self.MAX_ITERATIONS):  # Nifty, innit?
            print(
                f"Starting simulation nr. {i + 1}/{self.MAX_ITERATIONS}.. "
                f"Q-table size: {self.q_table.size}, shape: {self.q_table.shape}")

            self.rob.set_phone_tilt(np.pi / 6, 100)

            # A simulation runs until valid_environment returns False.
            while self.valid_environment():
                # Check in what state rob is, return tuple e.g. (0, 0, 0, 1, 0). A state is defined by rob's sensors.
                curr_state = self.handle_state()

                # Do we perform random action (due to epsilon < 1) or our best possible action?
                best_action = self.determine_action(curr_state)

                # Given our selected action (whether best or random), perform this action and update the Q-table.
                reward = self.update_q_table(best_action, curr_state)

                self.stats.add_reward(i, self.iteration_counter, reward)  # Add the reward for visualization purposes.
                self.iteration_counter += 1  # Keep track of how many actions this simulation does.

            self.stats.add_step_counter(i, self.iteration_counter)
            self.iteration_counter = 0

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

    def best_action_for_state(self, state):
        # Given a state (tuple format), what is the best action we take, i.e. for which action is the Q-value highest?
        q_row = self.q_table[state]
        max_val_indices = [i for i, j in enumerate(q_row) if j == max(q_row)]
        best_action = random.choice(max_val_indices) if len(max_val_indices) > 1 else np.argmax(q_row)

        return best_action

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
        # This function checks whether the current simulation can continue or not, depending on (several) criteria.
        c1 = self.iteration_counter >= self.MAX_SIMULATION_ITERATIONS

        return False if any([c1]) else True

    def initialize_q_table(self):
        # Initialize Q-table for states * action pairs with default values (0).
        # E.g. the size (5, 5, 5, 5, 5) denotes each sensor, with its amount of possible states (see func handle_state).
        return np.random.uniform(low=6, high=6, size=([2, 2, 2, 2, 2, 2] + [len(self.action_space)]))

    def initialize_q_table_prey(self):
        # Initialize Q-table for states * action pairs with default values (0).
        # E.g. the size (5, 5, 5, 5, 5) denotes each sensor, with its amount of possible states (see func handle_state).
        return np.random.uniform(low=6, high=6, size=([2, 2, 2, 2, 2, 2, 2, 2] + [len(self.action_space)]))

    def handle_state(self):
        contours_left_far, contours_left_close, contours_center_far, contours_center_close, contours_right_far, contours_right_close = self.determine_food()
        res = tuple()

        if contours_left_far > 0:
            res += (1,)
        else:
            res += (0,)
        if contours_left_close > 0:
            res += (1,)
        else:
            res += (0,)

        if contours_center_far > 0:
            res += (1,)
        else:
            res += (0,)
        if contours_center_close > 0:
            res += (1,)
        else:
            res += (0,)

        if contours_right_far > 0:
            res += (1,)
        else:
            res += (0,)
        if contours_right_close > 0:
            res += (1,)
        else:
            res += (0,)

        # print(f'State: {res}')
        return res

    def determine_food(self):
        image = self.rob.get_image_front()

        # Chop image vertically in two
        image_left_far = image[0:64, 0:30, :]
        image_left_close = image[64:128, 0:30, :]

        image_center_far = image[0:64, 30:98, :]
        image_center_close = image[64:128, 30:98, :]

        image_right_far = image[0:64, 98:128, :]
        image_right_close = image[64:128, 98:128, :]

        # Create mask for color green
        mask_left_far = cv2.inRange(image_left_far, (0, 0, 100), (140, 140, 255))
        mask_left_close = cv2.inRange(image_left_close, (0, 0, 100), (140, 140, 255))

        mask_center_far = cv2.inRange(image_center_far, (0, 0, 100), (140, 140, 255))
        mask_center_close = cv2.inRange(image_center_close, (0, 0, 100), (140, 140, 255))

        mask_right_far = cv2.inRange(image_right_far, (0, 0, 100), (140, 140, 255))
        mask_right_close = cv2.inRange(image_right_close, (0, 0, 100), (140, 140, 255))

        # Find contours, if present
        contours_left_far, _ = cv2.findContours(mask_left_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_left_close, _ = cv2.findContours(mask_left_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours_center_far, _ = cv2.findContours(mask_center_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_center_close, _ = cv2.findContours(mask_center_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours_right_far, _ = cv2.findContours(mask_right_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_right_close, _ = cv2.findContours(mask_right_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return len(contours_left_far), len(contours_left_close), len(contours_center_far), len(
            contours_center_close), len(contours_right_far), len(contours_right_close)

    def handle_action(self, action, curr_state):
        # This function should accept an action (0, 1, 2...) and move the robot accordingly (left, right, forward).
        # It returns two things: new_state, which is the state (in tuple format) after this action has been performed.
        # and reward, which is the reward from this action.

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

        self.rob.move(left, right, duration)

        reward = self.determine_reward(action, curr_state)
        return self.handle_state(), reward  # New_state, reward

    @staticmethod
    def determine_reward(action, curr_state):
        # This function determines the reward an action should get
        # Actions: left, right, forward, rright, lleft
        reward = 0

        if curr_state[1] > 0 or curr_state[3] > 0 or curr_state[5] > 0:
            # print('prey is in center, close')
            # Block is in center, close
            if curr_state[3] > 0:
                if action == 2:  # If block is center, reward forward action
                    reward += 1
                else:
                    reward -= 5  # And punish other actions

            else:  # Block is either left close or right close
                # print('prey is left/right close')
                if action == 0 and curr_state[1] > 0:  # We should do a small turn, so action 0 or 1
                    reward += 1
                elif action == 1 and curr_state[5] > 0:
                    reward += 1
                else:  # And if we don't, punish
                    reward -= 5

        elif curr_state[0] > 0 or curr_state[2] > 0 or curr_state[4] > 0:  # There are no close blocks, only far
            if curr_state[2] > 0:  # If center far is a block, move forward
                # print('prey is far center')
                if action == 2:
                    reward += 1
                else:
                    reward -= 5
            else:  # If left/right far is a block and not in center, turn slightly (action 0 or 1)
                # print('prey is far left/right')
                if action == 0 and curr_state[0] > 1:
                    reward += 1
                elif action == 1 and curr_state[4] > 0:
                    reward += 1
                else:
                    reward -= 5  # If we don't slightly turn, punish

        else:  # No block at all
            # print('no prey at all')
            if action == 3:  # We see nothing, so hard turn
                reward += 1
            else:  # If we don't do a hard turn right, punish
                reward -= 5
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
            env.stats.save_rewards(filename_rewards)
            env.store_q_table(filename_q_table)

    else:
        env.start_environment()

        # Save all data (rewards, food collected, steps done per simulation, and the Q-table.
        env.stats.save_rewards(
            f"results/reward_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle")
        env.store_q_table(
            f"results/q_table_data_{env.MAX_ITERATIONS}_{env.MAX_SIMULATION_ITERATIONS}_{EXPERIMENT_NAME}.pickle")


if __name__ == "__main__":
    main()