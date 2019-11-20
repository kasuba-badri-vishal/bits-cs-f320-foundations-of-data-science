import LinearModel
import DataUtils

import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop(columns="OSM_ID")    # drop unrequired feature
# normalize data
dataset = DataUtils.normalize(dataset, type="normal")
# split data
train, test = DataUtils.data_split(dataset, split_at=0.80)
# perform feature(x), target(y) split
x_train, y_train = DataUtils.xy_split(train, target="ALTITUDE")
x_test, y_test   = DataUtils.xy_split(test, target="ALTITUDE")
x_test.insert(0, "Const", np.ones(x_test.shape[0]))

st_time = time()
# w       = LinearModel.fit(x_train, y_train, alpha=3.5e-6, epsilion=1, method="SGD")
print(f"\nTime for Normal Gradient Descent: {time()-st_time}")

# print(w)
# print('Train Error(MSE):\t', LinearModel.error(w, x_train.values, y_train.values))
# print('Test Error(MSE):\t', LinearModel.test(w, x_test, y_test))
# print('Train Error(R2):\t', LinearModel.r2_error(w, x_train, y_train))
# print('Test Error(R2):\t\t', LinearModel.r2_error(w, x_test, y_test))

st_time = time()
w_list, lamdas, val_errs, train_errs = LinearModel.reg_fit(train, alpha=2e-7, epsilion=1e-1, method="L1GD")
print(f"\nTime for Regularized Gradient Descent: {time()-st_time}")

test_errs = list()
for w in w_list:
    test_errs.append( LinearModel.test(w, x_test, y_test) )

plt.title(f'Errors w.r.t lambda - Linear regression')
plt.ylabel('Loss')
plt.xlabel('Lambda')
plt.plot(lamdas, val_errs, label="validation error")
# plt.plot(lamdas, train_errs, label="training error")
# plt.plot(lamdas, test_errs, label="testing error")
plt.legend()
plt.grid(True)
plt.show()