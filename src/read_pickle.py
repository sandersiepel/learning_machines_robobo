import pickle
import numpy as np

with open('results/test/collision_data_5_100_test_1.pickle', 'rb') as fp:
    q_table = pickle.load(fp)
print(q_table)