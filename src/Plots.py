from Statistics import Statistics, Experiment
import numpy as np

# stats1 = Statistics(50, 200)
# name1 = "steady_epsilon"
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

exp1 = Experiment('static_epsilon', 50, 200, window=True, window_size=5)
exp2 = Experiment('dynamic_epsilon', 50, 200, window=True, window_size=5)
exp1.plot_two_experiments(exp2)

