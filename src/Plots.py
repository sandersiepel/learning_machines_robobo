from Statistics import Statistics, Experiment
import numpy as np

stats1 = Statistics(5, 100)
#
# stats1.read_data("train_week2")
stats1.read_food_amount("train_week2")
stats1.plot_data(stats1.collision)
# stats1.plot_two_different_axis(stats1.get_average_reward_simulation(), stats1.collision)

# stats2 = Statistics(50, 200)
# name2 = "dynamic_epsilon"
#
# stats1.read_data(name1)
# stats2.read_data(name2)
# stats1.plot_two_same_axis(stats1.get_average_reward_simulation(), stats1.get_data_rolling_window(5), opacity_n=1)
# stats2.plot_two_same_axis(stats2.get_average_reward_simulation(), stats2.get_data_rolling_window(5), opacity_n=1)
# stats1.plot_two_same_axis(stats1.get_data_rolling_window(5), stats2.get_data_rolling_window(5),
#                          title="Comparing a dynamic epsilon to a steady epsilon. ",
#                          label1="avg reward with epsilon = 0.9 \n using an average window of size 5",
#                          label2="avg reward with a dynamic epsilon \n using an average window of size 5")
# epsilon = list(np.arange(0.6, 0.99, 0.0078))
# stats2.plot_two_different_axis(stats2.get_average_reward_simulation(), epsilon, label1="average reward", label2="epsilon")

# exp1 = Experiment('train_week1', 100, 500, window=False, num_experiments=3)
# exp1.plot_single_experiment(title="Old actions")
# exp1.plot_reward_collision()
# exp2 = Experiment('old_actions', 50, 250, window=True, window_size=5, num_experiments=4)
# exp1.plot_two_experiments(exp2)

# stats = Statistics(20, 2)
# # stats.read_fitness("best_weights_20.pickle")
# stats.read_fitness("EC_fitness")
# stats.plot_two_same_axis(stats.rewards[:, 0], stats.rewards[:, 1], label1="max_reward", label2="avg_reward",
#                          x_label="Generation", title="Average and maximum fitness with EC", y_label="Fitness")

