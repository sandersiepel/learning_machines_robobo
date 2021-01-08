from Statistics import Statistics

stats = Statistics(50, 200)

stats.read_data()
data1 = stats.get_average_reward_simulation()
# stats.plot_data(stats.get_average_reward_simulation())
# stats.plot_data(stats.get_data_rolling_window(5))
stats.plot_two_same_axis(stats.get_average_reward_simulation(), stats.get_data_rolling_window(5),
                         title="Reward per simulation with a steady epsilon",
                         label1="avg reward", label2="avg reward with a window of size 5", opacity_n=1)