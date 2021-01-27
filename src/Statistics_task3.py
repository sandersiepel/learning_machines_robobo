import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import copy
import pandas as pd
import seaborn as sns


class StatisticsTask3():
    def __init__(self, max_simulations, max_iteration):
        self.simulations = max_simulations
        self.catch_score = np.zeros(max_simulations)
        self.steps = np.full(max_simulations, max_iteration)

        self.reward_predator = np.zeros(max_simulations)
        self.reward_prey = np.zeros(max_simulations)

    def catched(self, simulation, iteration):
        self.catch_score[simulation] = 1
        self.steps = iteration

    def add_rewards(self, simulation, reward_predator, reward_prey):
        self.reward_predator[simulation] = reward_predator
        self.reward_prey[simulation] = reward_prey

    def save_data(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(f"reward_prey_{name}", 'wb') as fp:
            pickle.dump(self.reward_prey, fp)

        with open(f"reward_predator_{name}", 'wb') as fp:
            pickle.dump(self.reward_predator, fp)

        with open(f"steps_{name}", 'wb') as fp:
            pickle.dump(self.steps, fp)

        with open(f"cathed_{name}", 'wb') as fp:
            pickle.dump(self.catch_score, fp)