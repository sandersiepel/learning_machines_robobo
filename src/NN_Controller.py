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
from Population import Controller, Population, Individual
from TrainQLearning import Environment


class Direction:
    LEFT = (-15, 15, 300)  # Action: 0, left
    RIGHT = (15, -15, 300)  # Action: 1, right
    FORWARD = (25, 25, 300)  # Action: 2, forward
    RRIGHT = (25, -25, 300)  # Action: 3, strong right
    LLEFT = (-25, 25, 300)  # Action: 4, strong left


class ECEnvironment:
    # All of our constants, prone to change.
    MAX_STEPS = 200  # Amount of actions within one simulation. Actions = Q-table updates.
    EXPERIMENT_NAME = 'best_weights_10'
    hostname = socket.gethostname()
    IP_ADDRESS = socket.gethostbyname(hostname)
    POP_SIZE = 10
    GEN_SIZE = 20

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)
        self.con = Controller()

    def train_ec(self):
        pop = Population(self.POP_SIZE, self.EXPERIMENT_NAME)
        pop.create_new()

        stats = Statistics(max_simulation=self.GEN_SIZE, max_iteration=2)

        for j in trange(self.GEN_SIZE):  # For every generation
            for i in range(self.POP_SIZE):  # For every individual
                self.rob.wait_for_ping()
                self.rob.play_simulation()
                self.pos = self.rob.position()

                fitness = self.eval_ind(pop.pop_list[i])
                pop.pop_list[i].fitness = fitness # evaluate an individual and store its fitness
                print("Fitness: " + str(fitness))

                self.rob.stop_world()
                self.rob.wait_for_ping()
            avg_fitness = pop.calculate_avg_fitness()
            best_fitness = pop.calculate_best_fitness()
            pop.print_fitness()
            stats.add_fitness(best_fitness, avg_fitness, j)
            pop.next_gen()
            # print(f"Generation {j+1}/{self.GEN_SIZE} avg: {pop.avg_fitness}, max: {pop.best_fitness}")
        stats.save_rewards("EC_fitness")

    def terminate_program(self, test1, test2):
        sys.exit(1)

    def eval_ind(self, ind):
        total_fitness = 0

        for i in range(self.MAX_STEPS):  # for all steps
            try:
                sensor_values = np.log(np.array(self.rob.read_irs())) / 10
            except:
                sensor_values = [0,0,0,0,0,0,0,0]

            sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
            sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

            out = self.con.forward(sensor_values, ind.weights)  # get the output of the NN

            self.do_action(out)  # Do the action
            best_action = np.argmax(out)
            # print(self.collision())
            # print("step: " + self.collision() + " ; " + str(best_action))
            reward = self.determine_reward(self.collision(), best_action)
            total_fitness += reward

        return total_fitness

    @staticmethod
    def determine_reward(collision, action):
        # This function determines the reward an action should get, depending on whether or not rob is about to
        # collide with an object within the environment.
        reward = 0

        if action in [0, 1]:  # Action is moving either left or right.
            if collision == "nothing":
                reward -= 0
            elif collision == "far":
                reward += 1
            elif collision == "close":
                reward += 1
        elif action in [3, 4]:
            if collision == "nothing":
                reward -= 0
            elif collision == "far":
                reward += 1
            elif collision == "close":
                reward += 1
        elif action == 2:  # Action is moving forward.
            if collision == "far":
                reward -= 0
            elif collision == "close":
                reward -= 1
            elif collision == "nothing":
                reward += 3

        return reward

    def do_action(self, out):
        best_action = np.argmax(out)
        if best_action == 0:
            left, right, duration = Direction.LEFT  # Left, action 0
        elif best_action == 1:
            left, right, duration = Direction.RIGHT  # Right, action 1
        elif best_action == 3:
            left, right, duration = Direction.RRIGHT  # Extreme right, action 3
        elif best_action == 4:
            left, right, duration = Direction.LLEFT  # Extreme left, action 4
        else:
            left, right, duration = Direction.FORWARD  # Forward, action 2
        self.rob.move(left, right, duration)

    def calc_distance(self):
        old_x = self.pos[0]
        old_y = self.pos[1]
        new_x = self.rob.position()[0]
        new_y = self.rob.position()[1]

        abs_x = abs(old_x - new_x)
        abs_y = abs(old_y - new_y)

        return np.sqrt((abs_x ** 2) + (abs_y ** 2))


    def collision(self):
        # This function checks whether rob is close to something or not. It returns the "distance", either "close", "far" or "nothing".
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        try:
            sensor_values = self.rob.read_irs()  # Should be absolute values (no log or anything).
        except:
            sensor_values = [1, 1, 1, 1, 1, 1, 1, 1]

        collision_far = any([0.13 <= i < 0.2 for i in sensor_values])
        collision_close = any([0 < i < 0.13 for i in sensor_values])

        if collision_close:
            # self.collision_counter += 1
            return "close"
        elif collision_far:
            return "far"
        else:
            # self.collision_counter = 0
            return "nothing"

    @staticmethod
    def check_forward(output):
        diff = abs(output[0] - output[1])
        if diff < 1:
            speed = output[0]
            if speed < 0:
                return 0.5 * speed * (1 - diff)
            else:
                return speed * (1 - diff)
        else:
            return -1


if __name__ == "__main__":
    env = ECEnvironment()
    env.train_ec()

