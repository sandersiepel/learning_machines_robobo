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
import pprint

# New comment TEST 2

class Direction:
    #testtesttest
    LEFT = (-25, 25, 300)  # Action: 0
    RIGHT = (25, -25, 300)  # Action: 1
    FORWARD = (25, 25, 300)  # Action: 2
    RRIGHT = (50, -50, 300)  # Action: 3
    LLEFT = (-50, 50, 300)  # Action: 4


class Environment:
    # All of our constants, prone to change.
    MAX_ITERATIONS = 10_000  # Amount of simulations until termination.
    MAX_SIMULATION_ITERATIONS = 500  # Amount of actions within one simulation. Actions = Q-table updates.
    LEARNING_RATE = .1
    DISCOUNT_FACTOR = .95

    EPSILON_LOW = .6  # Start epsilon value. This gradually increases.
    EPSILON_HIGH = .99  # End epsilon value
    EPSILON_INCREASE = .01  # How much should we increase the epsilon value with, each time?

    COLLISION_THRESHOLD = 100  # After how many collision actions should we reset the environment? Prevents rob getting stuck.

    action_space = [0, 1, 2, 3, 4]  # All of our available actions. Find definitions in the Direction class.
    collision_boundary = .1  # The boundary that determines collision or not.
    collision_counter, iteration_counter, epsilon_counter = 0, 0, 0

    # The epsilon_increase determines when the epsilon should be increased. This happens gradually from EPSILON_LOW
    # to EPSILON_HIGH during the amount of allowed iterations. So when MAX_ITERATIONS reaches its limit, so does
    # the epsilon value.
    epsilon_increase = int(((MAX_ITERATIONS * MAX_SIMULATION_ITERATIONS) // (EPSILON_HIGH - EPSILON_LOW) * 100) / 10_000)

    def __init__(self, new_QTable):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address='192.168.1.3', port=19997)
        if new_QTable:
            self.q_table = self.initialize_q_table()
        else:
            self.q_table = self.read_q_table()

    def read_q_table(self):
        "dit is een comment"
        "TODO Dit moeten we ook nog doen blablabla"
        with open('q_table', 'rb') as fp:
            q_table = pickle.load(fp)

            # hier comment
            # extra comment
            # extra comment
            print(3+5)
            print(3+5+9)
        return q_table

    def store_q_table(self):
        with open('q_table', 'wb') as fp:
            pickle.dump(self.q_table, fp)

    def start_environment(self):
        for i in range(1, self.MAX_ITERATIONS):
            print(f"Starting simulation nr. {i}")

            self.rob.play_simulation()
            # A simulation runs until valid_environment returns False.
            while self.valid_environment():
                curr_state = self.handle_state()  # Check in what state rob is, return tuple e.g. (0, 0, 0, 1, 0)

                # Do we perform random action (due to epsilon < 1) or our best possible action?
                if random.random() < (1 - self.EPSILON_LOW):
                    best_action = random.choice(self.action_space)
                else:
                    best_action = np.argmax(self.q_table[curr_state])

                # Given our selected action (whether best or random), perform this action and update the Q-table.
                self.update_q_table(best_action, curr_state)
                self.change_epsilon()  # Check if we should increase epsilon or not.
                self.iteration_counter += 1  # Keep track of how many actions this simulation does.
            else:
                print(f"Environment is not valid anymore, starting new environment")
                self.iteration_counter = 0
                self.rob.stop_world()

    def terminate_program(self, test1, test2):
        print("Ctrl-C received, terminating program")
        self.store_q_table()
        sys.exit(1)

    def valid_environment(self):
        # This function checks whether the current simulation can continue or not, depending on several criteria.
        if self.collision_counter > self.COLLISION_THRESHOLD:
            return False
        if self.iteration_counter >= self.MAX_SIMULATION_ITERATIONS:
            return False

        return True

    def change_epsilon(self):
        # This function changes the epsilon value if needed. Only does so if we did x amount of iterations, and the
        # current epsilon value is smaller than the epsilon limit (EPSILON_HIGH).
        if self.epsilon_counter == self.epsilon_increase:
            if self.EPSILON_LOW < self.EPSILON_HIGH:
                self.EPSILON_LOW += self.EPSILON_INCREASE
                print(f"Increasing epsilon to {self.EPSILON_LOW}")
                self.epsilon_counter = 0
        else:
            self.epsilon_counter += 1

    def initialize_q_table(self):
        # Initialize Q-table for states * action pairs with default values (0).
        # Since observation space is very large, we need to trim it down (bucketing) to only a select amount of
        # possible states, e.g. 4 for each sensor (4^8 = 65k). Or: use less sensors (no rear sensors for task 1).
        # The size (5, 5, 5, 5, 5) denotes each sensor, with its amount of possible states (see func handle_state).
        q_table = np.random.uniform(low=0, high=0, size=([5, 5, 5, 5, 5] + [len(self.action_space)]))
        # q_table[(0, 0, 0, 0, 0)][2] = 1  # Initialize the table with a forward move. Most likely not necessary.
        return np.round(q_table)

    def handle_state(self):
        # This function should return the values with which we can index our q_table, in tuple format.
        # So, it should take the last 5 sensor inputs (current state), transform each of them into a bucket where
        # the bucket size is already determined by the shape of the q_table.
        sensor_values = np.log(np.array(self.rob.read_irs())[3:]) / 10
        sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
        sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

        # Check what the actual sensor_values are (between [0, 1]) and determine their state
        indices = []
        for sensor_value in sensor_values:
            if 1 >= sensor_value >= 0.8:  # No need for action, moving forward is best.
                indices.append(0)
            elif 0.8 > sensor_value >= 0.6:
                indices.append(1)
            elif 0.6 > sensor_value >= 0.4:  # We see an object, but not really close yet.
                indices.append(2)
            elif 0.4 > sensor_value >= 0.2:
                indices.append(3)
            elif 0.2 > sensor_value >= 0:  # Close proximity.
                indices.append(4)

        # Return the values in tuple format, with which we can index our Q-table. This tuple is a representation
        # of the current state our robot is in (i.e. what does the robot see with its sensors).
        return tuple(indices)

    def handle_action(self, action):
        # This function should accept an action (0, 1, 2...) and move the robot accordingly (left, right, forward).
        # It returns two things: new_state, which is the state (in tuple format) after this action has been performed.
        # and reward, which is the reward from this action.
        reward = -1

        # Keep track of collision.
        collision = self.collision()

        if action == 0:
            left, right, duration = Direction.LEFT  # Left, action 0
            # self.rob.move(-10, -10, 50)
            # if collision:
            #     reward += 1
            # else:
            #     reward -= 1
        elif action == 1:
            left, right, duration = Direction.RIGHT  # Right, action 1
            # self.rob.move(-10, -10, 50)
            # if collision:
            #     reward += 1
            # else:
            #     reward -= 1
        elif action == 3:
            left, right, duration = Direction.RRIGHT  # Extreme right, action 3
            # self.rob.move(-10, -10, 50)
            # if collision:
            #     reward += 1
            # else:
            #     reward -= 1
        elif action == 4:
            left, right, duration = Direction.LLEFT  # Extreme left, action 4
            # self.rob.move(-10, -10, 50)
            # if collision:
            #     reward += 1
            # else:
            #     reward -= 1
        else:
            left, right, duration = Direction.FORWARD  # Forward, action 2
            if collision:
                reward -= 4
            else:
                reward += 2

        self.rob.move(left, right, duration)

        return self.handle_state(), reward  # New_state, reward

    def collision(self):
        # This function checks whether rob is close to something or not. If it's close (about to collide), return True
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        sensor_values = np.array(self.rob.read_irs())[3:]
        collision = any([0 < i < self.collision_boundary for i in sensor_values])

        if collision:
            self.collision_counter += 1
            return True
        else:
            self.collision_counter = 0
            return False

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

        print(f"Current state: {curr_state} with Q-values: {self.q_table[curr_state]}. \n"
              f"If we play our best action {best_action}, we end up in new state: {new_state} with Q-values: {self.q_table[new_state]}.\n"
              f"We now receive reward {reward}, the Q-value for current state is updates from {current_q} to {new_q}\n"
              f"The max future reward is: {max_future_q}\n")

        # And lastly, update the value in the Q-table.
        self.q_table[curr_state][best_action] = new_q


def main():
    env = Environment(True)
    env.start_environment()


if __name__ == "__main__":
    main()
