"""
This module implements solving normal equations
"""

import numpy as np

def run(x_train, y_train, function, error, alpha, epsilion=1E-9):

    print("Solving Normal Equations...")
    # alpha not used in normal eqns
    # insert column for x^0 or const factor x0 at starting of dataset (index = 0)
    x_train.insert(0, "Const", np.ones(x_train.shape[0]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # w = (X.transpose() * X).inverse() * X.transpose() * Y
    w = x_train.transpose().dot(x_train)
    w = np.linalg.inv(w)
    w = w.dot(x_train.transpose()).dot(y_train)

    return w