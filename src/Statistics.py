import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import copy


class Statistics:
    def __init__(self, max_simulation=0, max_iteration=0):
        """
        Saves all rewards (for each iteration in each simulation) while training. Both parameters are initialized
        during the training phase.
        The class also allows for some calculations and the generations of multiple different plots.

        Parameters:
            max_simulation: maximum amount of simulations
            max_iteration: maximum amount of iterations per simulation
        """
        self.rewards = np.random.uniform(low=0, high=0, size=(max_simulation, max_iteration))
        self.max_simulation = max_simulation
        self.max_iteration = max_iteration

    def read_data(self, name):
        # This function is used to open a pickle file (the rewards from a training session) and load it.
        with open(f'results/reward_data_{self.max_simulation}_{self.max_iteration}_{name}.pickle', 'rb') as fp:
            rewards = pickle.load(fp)
        self.rewards = rewards

    def save_rewards(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(name, 'wb') as fp:
            pickle.dump(self.rewards, fp)

    def add_reward(self, simulation, iteration, reward):
        self.rewards[simulation][iteration] = reward

    def get_rewards(self):
        return self.rewards

    def get_average_reward_simulation(self):
        avg_reward = []
        for i in range(len(self.rewards)):
            avg_reward.append(np.mean(self.rewards[i]))
        return avg_reward

    def get_total_reward_simulation(self):
        sum_reward = []
        for i in range(len(self.rewards)):
            sum_reward.append(np.sum(self.rewards[i]))
        return sum_reward

    def get_data_rolling_window(self, window_size):
        """
        Applies the average value to a rolling window

        Parameters:
            The size of the window. This value must be uneven!
        """

        data = self.get_average_reward_simulation()
        padded_data = copy.deepcopy(data)
        window_data = []
        pad_size = (window_size - 1) // 2
        last_value = data[-1]

        for i in range(pad_size):
            padded_data = np.insert(padded_data, 0, 0)
            padded_data = np.insert(padded_data, -1, last_value)

        for i in range(len(data)):
            total = 0
            for j in range(window_size):
                total += padded_data[i + j]
            window_data.append(total/window_size)

        return window_data

    @staticmethod
    def plot_data(data, y_label="reward", x_label="Simulation number", title="reward per simulation"):
        """
        Plots a single data list
        """
        plt.plot(data)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_two_same_axis(data1, data2, y_label="Reward", x_label="Simulation number",
                           title="Reward per simulation", label1="label1", label2="label2", opacity_n=0):
        """
        Plots two data sets in the same plot using the same y-axis
        """
        if opacity_n == 1:
            plt.plot(data1, label=label1, alpha=0.7)
            plt.plot(data2, label=label2)
        elif opacity_n == 2:
            plt.plot(data2, label=label2, alpha=0.7)
            plt.plot(data1, label=label1)
        else:
            plt.plot(data1, label=label1)
            plt.plot(data2, label=label2)

        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_two_different_axis(data1, data2, y_label1="Reward", y_label2="Epsilon",
                                x_label="Simulation number",
                                title="Reward per simulation", label1="label1", label2="label2"):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label1)
        lns1 = ax1.plot(data1, color='tab:blue', label=label1)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(y_label2)  # we already handled the x-label with ax1
        lns2 = ax2.plot(data2, color='tab:orange', label=label2)
        ax2.tick_params(axis='y')

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0)

        plt.title(title)
        plt.show()