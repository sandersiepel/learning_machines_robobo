import pickle
import numpy as np

with open('q_table', 'rb') as fp:
    q_table = pickle.load(fp)
print(q_table)