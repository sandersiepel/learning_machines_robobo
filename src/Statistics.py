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

        Parameters:
            max_simulation: maximum amount of simulations
            max_iteration: maximum amount of iterations per simulation
        """
        self.rewards = np.random.uniform(low=0, high=0, size=(max_simulation, max_iteration))
        self.max_simulation = max_simulation
        self.max_iteration = max_iteration

    def read_data(self):
        # This function is used to open a pickle file (the rewards from a training session) and load it.
        with open(f'results/reward_data_{self.max_simulation}_{self.max_iteration}.pickle', 'rb') as fp:
            rewards = pickle.load(fp)
        self.rewards = rewards

    def save_rewards(self):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(f'src/results/reward_data_{self.max_simulation}_{self.max_iteration}.pickle', 'wb') as fp:
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

    def plot_data(self, data, ylabel="reward", xlabel = "Simulation number", title = "reward per simulation"):
        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.show()
