import matplotlib.pyplot as plt
import numpy as np
import pickle

class Statistics:
    def __init__(self, max_simulation, max_iteration):
        """
        Saves all rewards while training

        Parameters:
            max_simulation: maximum amount of simulations
            max_iteration: maximum amount of iterations per simulation
        """
        self.rewards = np.random.uniform(low=0, high=0, size=(max_simulation, max_iteration))

    def read_data(self):
        with open('data', 'rb') as fp:
            rewards = pickle.load(fp)
        self.rewards = rewards

    def store_data(self):
        with open('data', 'wb') as fp:
            pickle.dump(self.rewards, fp)

    def add_reward(self, simulation, iteration, reward):
        self.rewards[simulation][iteration] = reward

    def get_rewards(self):
        return self.rewards

    def get_average_reward(self):
        avg_reward = []
        for i in range(len(self.rewards)):
            avg_reward.append(np.mean(self.rewards[i]))
        return avg_reward

    def get_total_reward(self):
        sum_reward = []
        for i in range(len(self.rewards)):
            sum_reward.append(np.sum(self.rewards[i]))
        return sum_reward

    def plot(self, data, y_label, x_label, titel):
        plt.plot(data)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(titel)
        plt.show()