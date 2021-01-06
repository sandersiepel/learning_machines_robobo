from Statistics import Statistics

stats = Statistics(10, 50)

stats.read_data()
data = stats.get_average_reward()
stats.plot(data, "xas", "yas", "titel")