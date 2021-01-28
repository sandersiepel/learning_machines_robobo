from Statistics import Statistics, Experiment
from Statistics_task3 import StatisticsTask3
import numpy as np

# stats1 = Statistics(20, 200)
# stats2 = Statistics(20, 200)
# stats3 = Statistics(20, 200)
#
# stats1.read_step_counter("train3")
# stats1.read_food_amount("train3")
#
# stats2.read_step_counter("train4")
# stats2.read_food_amount("train4")
#
# stats3.read_step_counter("train6")
# stats3.read_food_amount("train6")
#
# stats1.plot_three_same_axis(stats1.food_amount, stats2.food_amount, stats3.food_amount, y_label="food amount",
#                             title="first vs second", label1="c=50, a=1/-1", label2="c=50, a=1/-5, df=0.8",
#                             label3="c=50, a=(1/-5) + dir, df=0.8")

# stats1.plot_data(stats1.food_amount, title="c=50, a=1/-1", y_label="food amount")

# stats1.plot_two_different_axis(stats1.food_amount, stats1.step_counter)


# stats1.plot_two_different_axis(stats1.get_average_reward_simulation(), stats1.collision)
# name = 'R3_ellipse_draw4'
# stats = Statistics(50, 250)
# stats.read_step_counter(name)
# stats.read_food_amount(name)
# stats.plot_data(stats.food_amount, title="only collsion", y_label="food amount")
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

#  Steady epsilon vs constant

# exp1 = Experiment('CRS_ellipse_draw4', 50, 250, window=True, num_experiments=2, label="Ellipse")
# exp2 = Experiment('train_8_mr', 50, 250, window=True, num_experiments=5, label="Squares")
# exp1.plot_single_experiment('food')
# exp1.plot_two_experiments(exp2, "food")

# exp2 = Experiment('train_8_mr', 50, 250, window=True, num_experiments=5)
# exp2.plot_single_experiment('steps')
# exp2.plot_single_experiment('steps')
# exp1 = Experiment('CRS_ellipse_draw4', 50, 250, window=True, num_experiments=3)
# exp1.plot_single_experiment('steps')
# exp1.plot_single_experiment('food')

# exp1.plot_two_experiments(exp2, ydata="steps")
# exp1.plot_two_experiments(exp2, ydata="steps",
#                           title="Comparing average amount of steps to gather all rewards \n with their std and a draw distance of 4")

# stats = Statistics(20, 2)
# # stats.read_fitness("best_weights_20.pickle")
# stats.read_fitness("EC_fitness")
# stats.plot_two_same_axis(stats.rewards[:, 0], stats.rewards[:, 1], label1="max_reward", label2="avg_reward",
#                          x_label="Generation", title="Average and maximum fitness with EC", y_label="Fitness")

stats = StatisticsTask3(50, 250)
stats.read_data("test")
stats.plot_two_different_axis()
# stats.plot_rewards()
# stats.plot_data(stats.catch_score)
# print(stats.steps)
# stats.plot_data(stats.steps)



