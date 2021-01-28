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
        self.steps[simulation] = iteration

    def add_rewards(self, simulation, reward_predator, reward_prey):
        self.reward_predator[simulation] = reward_predator
        self.reward_prey[simulation] = reward_prey

    def save_data(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(f"results/reward_prey_{name}", 'wb') as fp:
            pickle.dump(self.reward_prey, fp)

        with open(f"results/reward_predator_{name}", 'wb') as fp:
            pickle.dump(self.reward_predator, fp)

        with open(f"results/steps_{name}", 'wb') as fp:
            pickle.dump(self.steps, fp)

        with open(f"results/cathed_{name}", 'wb') as fp:
            pickle.dump(self.catch_score, fp)

    def read_data(self, name):
        with open(f"results/reward_prey_{name}", 'rb') as fp:
            reward_prey = pickle.load(fp)
        self.reward_prey = reward_prey

        with open(f"results/reward_predator_{name}", 'rb') as fp:
            reward_predator = pickle.load(fp)
        self.reward_predator = reward_predator

        with open(f"results/steps_{name}", 'rb') as fp:
            steps = pickle.load(fp)
        self.steps = steps

        with open(f"results/cathed_{name}", 'rb') as fp:
            catch_score = pickle.load(fp)
        self.catch_score = catch_score

    def plot_rewards(self):
        plt.plot(self.reward_prey / self.steps, label="prey")
        plt.plot(self.reward_predator/ self.steps, label="predator")
        # plt.xlabel(x_label)
        # plt.title(title)
        plt.legend()
        plt.show()

    def plot_data(self, data):
        plt.plot(data)
        plt.show()

    def plot_two_different_axis(self, y_label1="Reward", y_label2="catched",
                                x_label="Simulation number",
                                title="Reward per simulation", label1="prey (left)", label2="predator (left)"):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label1)
        lns1 = ax1.plot(self.reward_prey, color='tab:blue', label=label1)
        lns2 = ax1.plot(self.reward_predator, color='tab:orange', label=label2)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(y_label2)  # we already handled the x-label with ax1
        lns3 = ax2.plot(self.catch_score, color='tab:green', label="catched (right)", alpha=0.7)
        ax2.tick_params(axis='y')

        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0)
        plt.title(title)
        plt.show()