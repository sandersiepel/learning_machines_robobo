from Statistics import Statistics

stats = Statistics(100, 200)

stats.read_data()
data1 = stats.get_average_reward_simulation()
# stats.plot_data(stats.get_average_reward_simulation())
# stats.plot_data(stats.get_data_rolling_window(5))
stats.plot_two_same_axis(stats.get_average_reward_simulation(), stats.get_data_rolling_window(5), label1="avg reward",
                         label2="avg reward with a window of size 5", opacity_n=1)