import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


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

    def plot_average_reward_simulation(self):
        plt.plot(self.get_average_reward_simulation())
        plt.ylabel("Average reward")
        plt.xlabel("Simulation number")
        plt.title("Average reward per simulation")
        plt.show()

    def plot_total_reward_simulation(self):
        plt.plot(self.get_total_reward_simulation())
        plt.ylabel("Total reward")
        plt.xlabel("Simulation number")
        plt.title("Total reward per simulation")
        plt.show()
