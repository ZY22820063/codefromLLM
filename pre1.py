import sys

sys.path.append("./utils")
from preprocessing import del_corr_mask, del_na_sd_mask
import pandas as pd, numpy as np

dataset = pd.read_csv("TS-origin-cut.csv",
                      encoding="utf-8")

X = dataset.iloc[:, 3:].values

# X = np.array(dataset.iloc[:, 2:], dtype=float) # ndarray
Y = dataset.iloc[:, 2].values.astype(float)  # ndarray.astype
columns = dataset.columns[3:]
classes = dataset.iloc[:, 1].values
numbers = dataset.iloc[:, 0].values

mask = del_na_sd_mask(X)
X = X[:, mask].astype(float)
columns = columns[mask]
# print("columns",columns)

# print("11111",type(X))
mask = del_corr_mask(X)
X = X[:, mask]
columns = columns[mask]
# print("mask",mask) 

new_data = np.concatenate([numbers.reshape(-1, 1), classes.reshape(-1, 1), Y.reshape(-1, 1), X], axis=1) # np.concatxxxx
new_data = pd.DataFrame(new_data)
new_data.columns = ["number", "class", "targets"] + columns.tolist()
# new_data.to_excel("2.xlsx", index=None)
new_data.to_csv("TS-pre-cut.csv", index=None)
