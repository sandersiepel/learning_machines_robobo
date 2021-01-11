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

class controller():
    N_HIDDEN = 10
    N_OUTPUT = 2

    def sigmoid(self, matrix):
        newValues = np.empty(matrix.shape)
        for i in range(len(newValues)):
            newValues[i] = 1/(1+np.exp(-matrix[i]))
        return newValues

    def forward(self, irs_input, weights):
        bias1 = weights[:self.N_HIDDEN].reshape(1, self.N_HIDDEN)

        weight1_slice = len(irs_input) * self.N_HIDDEN + self.N_HIDDEN
        weight1 = weights[self.N_HIDDEN:weight1_slice].reshape((len(irs_input), self.N_HIDDEN))

        bias2 = weights[weight1_slice:weight1_slice + self.N_OUTPUT].reshape(1, self.N_OUTPUT)
        weight2 = weights[weight1_slice + self.N_OUTPUT:].reshape((self.N_HIDDEN, self.N_OUTPUT))

        output1 = self.sigmoid(irs_input.dot(weight1) + bias1)
        output2 = output1.dot(weight2) + bias2

        return output2[0]


class Environment:
    # All of our constants, prone to change.
    MAX_STEPS = 100  # Amount of actions within one simulation. Actions = Q-table updates.
    EXPERIMENT_NAME = 'old_actions'
    hostname = socket.gethostname()
    IP_ADDRESS = socket.gethostbyname(hostname)
    POP_SIZE = 10

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)

    def start_environment(self):
        self.rob.wait_for_ping()
        self.rob.play_simulation()
        n_w = 112
        w = np.random.uniform(low=-1, high=1, size=(10, n_w))
        self.con = controller()

        for i in range(self.POP_SIZE):

            ind_fitness = self.eval_ind(w[i])
            print(ind_fitness)
            self.rob.stop_world()
            self.rob.wait_for_ping()


    def terminate_program(self, test1, test2):
        sys.exit(1)

    def eval_ind(self, w):
        collsion_count = 0
        total_speed = 0

        for i in trange(self.MAX_STEPS):
            sensor_values = np.log(np.array(self.rob.read_irs())) / 10

            sensor_values = np.where(sensor_values == -np.inf, 0, sensor_values)  # Remove the infinite values.
            sensor_values = (sensor_values - -0.65) / 0.65  # Scales all variables between [0, 1] where 0 is close proximity.

            out = self.con.forward(sensor_values, w)

            self.rob.move(out[0], out[1], 300)

            if self.collision():
                collsion_count += 1

            total_speed += self.check_forward(out)

        return -(collsion_count * 10) + total_speed


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

    def check_forward(self, output):
        if output[0] - output[1] == 0:
            speed = output[0]
            return speed
        else:
            return -1




env = Environment()
env.start_environment()
