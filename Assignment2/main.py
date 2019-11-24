import PolynomialModel
import DataUtils

import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")
dataset = dataset.drop(columns="OSM_ID")    # drop unrequired feature

degree  = 6     # use 1 for linear model
# method  = "L1GD"

# add polynomial features if degree > 1
dataset = PolynomialModel.transform_dataset(dataset, degree)
# normalize data
dataset = DataUtils.normalize(dataset, type="min-max")
# split data
train, test = DataUtils.data_split(dataset, split_at=0.80)
# perform feature(x), target(y) split
x_train, y_train = DataUtils.xy_split(train)
x_test, y_test   = DataUtils.xy_split(test)
# print(f"x_train dimensions: {x_train.shape}")

st_time = time()
print(f"Using polynomial of degree: {degree}")
w       = PolynomialModel.fit(x_train, y_train, alpha=2.23e-7, epsilion=1e-4, method="GD")
# w_list, lamdas, val_errs, train_errs = PolynomialModel.reg_fit(train, alpha=4e-6, epsilion=5e-4, method=method, degree=degree)

print(f"\nTime to find weights: {time()-st_time}")
x_test.insert(0, "Const", np.ones(x_test.shape[0]))
print(w)
print('Train Error(MSE):\t', PolynomialModel.error(w, x_train.values, y_train.values))
print('Test Error(MSE):\t', PolynomialModel.test(w, x_test, y_test))
print('Train Error(R2):\t', PolynomialModel.r2_error(w, x_train, y_train))
print('Test Error(R2):\t\t', PolynomialModel.r2_error(w, x_test, y_test))

# test_errs = list()
# for w in w_list:
#     test_errs.append( PolynomialModel.test(w, x_test, y_test) )

# plt.title(f'Errors w.r.t lambda - degree: {degree}, method: {method}')
# plt.ylabel('MSE')
# plt.xlabel('lambda')
# plt.plot(lamdas, val_errs, label="validation error")
# # plt.plot(lamdas, train_errs, label="training error")
# # plt.plot(lamdas, test_errs, label="testing error")
# plt.legend()
# plt.grid(True)
# plt.show()