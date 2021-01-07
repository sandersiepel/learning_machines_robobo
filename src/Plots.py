from Statistics import Statistics

stats = Statistics(500, 250)

stats.read_data()
data1 = stats.get_average_reward_simulation()
stats.plot_data(stats.get_average_reward_simulation())
stats.plot_data(stats.get_data_rolling_window(5))
