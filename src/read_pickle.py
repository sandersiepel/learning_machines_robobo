import pickle
import numpy as np

with open('results/q_table_5_100.pickle', 'rb') as fp:
    q_table = pickle.load(fp)
print(q_table)