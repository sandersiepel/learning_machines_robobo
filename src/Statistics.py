import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import copy
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


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
        self.collision = np.random.uniform(low=0, high=0, size=max_simulation)
        self.food_amount = np.random.uniform(low=0, high=0, size=max_simulation)
        self.step_counter = np.random.uniform(low=0, high=0, size=max_simulation)
        self.max_simulation = max_simulation
        self.max_iteration = max_iteration

    def read_data(self, name):
        # This function is used to open a pickle file (the rewards from a training session) and load it.
        with open(f'results/reward_data_{self.max_simulation}_{self.max_iteration}_{name}.pickle', 'rb') as fp:
            rewards = pickle.load(fp)
        self.rewards = rewards

    def read_fitness(self, name):
        # This function is used to open a pickle file (the rewards from a training session) and load it.
        with open(name, 'rb') as fp:
            rewards = pickle.load(fp)
        self.rewards = rewards

    def read_experiment(self, name, number):
        # This function is used to open a pickle file (the rewards from a training session) and load it.
        reward_data = f'results/{name}/reward_data_{self.max_simulation}_{self.max_iteration}_{name}_{number}.pickle'
        collision_data = f'results/{name}/collision_data_{self.max_simulation}_{self.max_iteration}_{name}_{number}.pickle'
        food_data = f'results/{name}/food_data_{self.max_simulation}_{self.max_iteration}_{name}_{number}.pickle'

        with open(reward_data, 'rb') as fp:
            rewards = pickle.load(fp)
        self.rewards = rewards

        with open(collision_data, 'rb') as fp:
            collision = pickle.load(fp)
        self.collision = collision

        with open(food_data, 'rb') as fp:
            food_data = pickle.load(fp)
        self.food_amount = food_data

    def add_step_counter(self, simulation, step_counter):
        self.step_counter[simulation] = step_counter

    def save_rewards(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(name, 'wb') as fp:
            pickle.dump(self.rewards, fp)

    def save_step_counter(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(name, 'wb') as fp:
            pickle.dump(self.step_counter, fp)

    def save_collision(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(name, 'wb') as fp:
            pickle.dump(self.collision, fp)

    def save_food_amount(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(name, 'wb') as fp:
            pickle.dump(self.food_amount, fp)

    def read_collision(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(f'results/collision_data_{self.max_simulation}_{self.max_iteration}_{name}.pickle', 'rb') as fp:
            collision = pickle.load(fp)
        self.collision = collision

    def read_step_counter(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(f'results/step_counter_{self.max_simulation}_{self.max_iteration}_{name}.pickle', 'rb') as fp:
            step_counter = pickle.load(fp)
        self.step_counter = step_counter

    def read_food_amount(self, name):
        # This function is used to save the accumulated rewards during training in a pickle file.
        with open(f'results/food_amount_{self.max_simulation}_{self.max_iteration}_{name}.pickle', 'rb') as fp:
            food_amount = pickle.load(fp)
        self.food_amount = food_amount

    def add_reward(self, simulation, iteration, reward):
        # This function adds a reward to the simulation object
        try:
            self.rewards[simulation][iteration] = reward
        except IndexError:
            print(f'Weird index error, should not happen. Indices: {simulation}/{iteration}, reward: {reward}')

    def add_collision(self, simulation, total_collision):
        self.collision[simulation] = total_collision

    def add_food(self, simulation, total_food):
        self.food_amount[simulation] = total_food

    def add_fitness(self, max_fitness, avg_fitness, generation):
        self.rewards[generation][0] = max_fitness
        self.rewards[generation][1] = avg_fitness

    def get_rewards(self):
        # This function returns all rewards
        return self.rewards

    def get_average_reward_simulation(self):
        # This function returns the average reward per simulation
        avg_reward = []
        for i in range(len(self.rewards)):
            avg_reward.append(np.mean(self.rewards[i]))
        return avg_reward

    def get_total_reward_simulation(self):
        # This function returns the total reward per simulation
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

        # data = self.get_average_reward_simulation()
        data = raw_data
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


class Experiment:

    def __init__(self, experiment_name, num_simulations, num_iterations, num_experiments=5, window=False,
                 window_size=5):
        """
        Reads multiple experiments into one dataframe
        """
        self.df = pd.DataFrame(columns=["Reward", "Run"])
        for i in range(num_experiments):
            stat = Statistics(num_simulations, num_iterations)
            stat.read_experiment(experiment_name, i+1)
            if window:
                df2 = pd.DataFrame(stat.get_data_rolling_window(window_size), columns=["Reward"])
            else:
                df2 = pd.DataFrame(stat.get_average_reward_simulation(), columns=["Reward"])
                # df2 = pd.DataFrame(stat.get_total_reward_simulation(), columns=["Reward"])
            df2["collision"] = stat.collision
            # df2["collision"] = stat.get_data_rolling_window(5)
            df2['Run'] = i+1
            df2['Simulation'] = df2.index + 1
            df2['experiment_name'] = experiment_name
            self.df = self.df.append(df2, ignore_index=True)

    def plot_single_experiment(self, title="Average over 5 runs with its std"):
        ax = sns.lineplot(data=self.df, x="Simulation", y="Reward")
        ax.set_title(title)
        plt.show()

    def plot_two_experiments(self, experiment, title="Comparing average rewards over 5 runs with their std"):
        df2 = self.df.append(experiment.df, ignore_index=True)
        ax = sns.lineplot(data=df2, x="Simulation", y="Reward", hue="experiment_name")
        ax.set_title(title)
        plt.show()

    def plot_reward_collision(self, title="rewards and collisions over training", label1=""):

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('Simulation', fontsize=16)
        ax1.set_ylabel('Reward', fontsize=16)
        lns1 = sns.lineplot(data=self.df, x="Simulation", y="Reward")

        ax2 = lns1

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Collision', fontsize=16)

        ax2 = sns.lineplot(data=self.df, x="Simulation", y="collision", color=color)

        orange = mpatches.Patch(color='tab:orange', label='Collision')
        blue = mpatches.Patch(color='tab:blue', label='Reward')
        plt.legend(handles=[orange, blue])

        ax2.tick_params(axis='y', color=color)

        plt.show()