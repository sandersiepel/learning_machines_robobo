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


class Environment:
    # All of our constants, prone to change.
    MAX_STEPS = 100  # Amount of actions within one simulation. Actions = Q-table updates.
    EXPERIMENT_NAME = 'old_actions'
    hostname = socket.gethostname()
    IP_ADDRESS = socket.gethostbyname(hostname)
    POP_SIZE = 20
    GEN_SIZE = 20

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)

    def start_environment(self):
        n_w = 112
        pop = Population(self.POP_SIZE)
        pop.create_new()
        self.con = Controller()

        for j in range(self.GEN_SIZE):
            for i in range(self.POP_SIZE):
                self.rob.wait_for_ping()
                self.rob.play_simulation()
                self.pos = self.rob.position()
                fitness = self.eval_ind(pop.pop_list[i])
                pop.pop_list[i].fitness = fitness

                self.rob.stop_world()
                self.rob.wait_for_ping()
            pop.next_gen()
            print(f"Generation {j+1}/{self.GEN_SIZE} avg: {pop.avg_fitness}, max: {pop.best_fitness}")


    def terminate_program(self, test1, test2):
        sys.exit(1)

    def eval_ind(self, ind):
        collision_count = 0
        total_speed = 0
        total_distance = 0
        total_fitness = 0

        for i in range(self.MAX_STEPS):

            sensor_values = np.log(np.array(self.rob.read_irs())) / 10
            sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
            sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

            out = self.con.forward(sensor_values, ind.weights)

            self.rob.move(out[0], out[1], 300)

            if self.collision():
                collision_count += 1

            total_speed += self.check_forward(out)

            total_distance += self.calc_distance()

            s_trans = abs(out[0] + abs(out[1]))
            s_rot = abs(out[0] - out[1]) / 200
            v_sens = np.min(sensor_values)
            total_fitness += s_trans * (1-s_rot) * v_sens

        # return -(collision_count * 10) + total_speed + total_distance
        return total_fitness

    def calc_distance(self):
        old_x = self.pos[0]
        old_y = self.pos[1]
        new_x = self.rob.position()[0]
        new_y = self.rob.position()[1]

        abs_x = abs(old_x - new_x)
        abs_y = abs(old_y - new_y)

        return np.sqrt((abs_x ** 2) + (abs_y ** 2))

    def collision(self):
        # This function checks whether rob is close to something or not. It returns True if it's about to collide with
        # another object. Also returns the "distance", either "close", "far" or "nothing".
        # It also keeps track of the collision counter. If this counter exceeds its threshold (COLLISION_THRESHOLD)
        # then the environment should reset (to avoid rob getting stuck).
        try:
            sensor_values = self.rob.read_irs()  # Should be absolute values (no log or anything).
        except RuntimeWarning:
            sensor_values = [0, 0, 0, 0, 0, 0, 0, 0]

        collision_far = any([0.13 <= i < 0.2 for i in sensor_values])
        collision_close = any([0 < i < 0.13 for i in sensor_values])

        if collision_close:
            return True

    @staticmethod
    def check_forward(output):
        if output[0] == output[1]:
            speed = output[0]
            return speed
        else:
            return -1


env = Environment()
env.start_environment()
