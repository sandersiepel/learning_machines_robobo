import pickle
import numpy as np
import pandas as pd

with open('results/q_table_data_50_200_train_week2.pickle', 'rb') as fp:
    q_table = pickle.load(fp)


# arrays = [["0", "0", "0", "0", "1", "1", "1", "1"],
#           ["0", "0", "1", "1", "0", "0", "1", "1"],
#           ["0", "1", "0", "1", "0", "1", "0", "1"]]
# tuples = list(zip(*arrays))
#
# index = pd.MultiIndex.from_tuples(tuples, names=["left", "center", "right"])
# s = pd.DataFrame(np.nan, index=index, columns=["left", "right", "Center", "hard left", "hard right"])
#
# for i in range(2):
#     for j in range(2):
#         for k in range(2):
#             s.loc[(str(i), str(j), str(k))] = q_table[(i, j, k)]
#
# print(s.head(8))

