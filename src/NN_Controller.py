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
    MAX_STEPS = 250  # Amount of actions within one simulation. Actions = Q-table updates.
    EXPERIMENT_NAME = 'old_actions'
    hostname = socket.gethostname()
    IP_ADDRESS = socket.gethostbyname(hostname)

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo().connect(address=self.IP_ADDRESS, port=19997)

    def start_environment(self):
        self.rob.wait_for_ping()
        self.rob.play_simulation()
        n_w = 112
        w = np.random.uniform(low=-1, high=1, size=n_w)
        con = controller()

        for i in trange(self.MAX_STEPS):  # Nifty, innit?

            irs_in = np.array(self.rob.read_irs())

            out = con.forward(irs_in, w)

            self.rob.move(out[0], out[1], 300)

    def terminate_program(self, test1, test2):
        sys.exit(1)




env = Environment()
env.start_environment()
