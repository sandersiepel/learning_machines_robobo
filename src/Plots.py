from Statistics import Statistics

stats = Statistics(10, 50)

stats.read_data()
data1 = stats.get_average_reward()
stats.plot(data1, "xas", "yas", "titel")