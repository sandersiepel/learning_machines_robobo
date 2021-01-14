import pickle
import numpy as np

with open('best_weights', 'rb') as fp:
    q_table = pickle.load(fp)
print(q_table)