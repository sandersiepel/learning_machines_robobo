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
    MAX_STEPS = 150  # Amount of actions within one simulation. Actions = Q-table updates.
    EXPERIMENT_NAME = 'old_actions'
    hostname = socket.gethostname()
    IP_ADDRESS = socket.gethostbyname(hostname)
    POP_SIZE = 15
    GEN_SIZE = 15

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)

    def start_environment(self):
        pop = Population(self.POP_SIZE)
        pop.create_new()
        self.con = Controller()

        stats = Statistics(max_simulation=self.GEN_SIZE, max_iteration=2)

        for j in trange(self.GEN_SIZE):
            for i in range(self.POP_SIZE):
                self.rob.wait_for_ping()
                self.rob.play_simulation()
                self.pos = self.rob.position()
                fitness = self.eval_ind(pop.pop_list[i])
                pop.pop_list[i].fitness = fitness

                self.rob.stop_world()
                self.rob.wait_for_ping()
            pop.next_gen()
            stats.add_fitness(pop.best_fitness, pop.avg_fitness, j)
            # print(f"Generation {j+1}/{self.GEN_SIZE} avg: {pop.avg_fitness}, max: {pop.best_fitness}")
        stats.save_rewards("EC_fitness")


    def terminate_program(self, test1, test2):
        sys.exit(1)

    def eval_ind(self, ind):
        collision_count = 0
        total_speed = 0
        total_distance = 0
        total_fitness = 0

        # act = Environment()

        for i in range(self.MAX_STEPS):

            sensor_values = np.log(np.array(self.rob.read_irs())) / 10
            sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
            sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

            out = self.con.forward(sensor_values, ind.weights)

            # self.rob.move(out[0], out[1], 300)


            self.do_action(out)
            best_action = np.argmax(out)
            # print(best_action)
            reward = Environment.determine_reward(self.collision(), best_action)
            total_fitness += reward
            # print(reward)

            # if self.collision():
            #     collision_count += 1

            # total_speed += self.check_forward(out)
            #
            # total_distance += self.calc_distance()
            #
            # s_trans = abs(out[0] + abs(out[1]))
            # s_rot = abs(out[0] - out[1]) / 200
            # v_sens = np.min(sensor_values)
            # total_fitness += s_trans * (1-s_rot) * v_sens

        # return -(collision_count * 10) + total_speed + total_distance
        return total_fitness

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

    # def collision(self):
    #     # This function checks whether rob is close to something or not. It returns True if it's about to collide with
    #     # another object. Also returns the "distance", either "close", "far" or "nothing".
    #     # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
    #     # then the environment should reset (to avoid rob getting stuck).
    #     try:
    #         sensor_values = self.rob.read_irs()  # Should be absolute values (no log or anything).
    #     except RuntimeWarning:
    #         sensor_values = [0, 0, 0, 0, 0, 0, 0, 0]
    #
    #     collision_far = any([0.13 <= i < 0.2 for i in sensor_values])
    #     collision_close = any([0 < i < 0.13 for i in sensor_values])
    #
    #     if collision_close:
    #         return True

    def collision(self):
        # This function checks whether rob is close to something or not. It returns the "distance", either "close", "far" or "nothing".
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        try:
            sensor_values = self.rob.read_irs()  # Should be absolute values (no log or anything).
        except:
            sensor_values = [0, 0, 0, 0, 0, 0, 0, 0]

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


env = ECEnvironment()
env.start_environment()
