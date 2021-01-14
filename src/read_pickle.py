import pickle
import numpy as np

with open('results/q_table_50_200.pickle', 'rb') as fp:
    q_table = pickle.load(fp)
print(q_table)