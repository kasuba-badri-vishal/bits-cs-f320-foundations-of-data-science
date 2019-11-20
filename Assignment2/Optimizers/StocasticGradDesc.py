"""
This module implements stocastic gradient descent
"""
import numpy as np
from tqdm import tqdm

f = None

def grad_f(w: np.array, x: np.array, y: np.array) -> int:
    res = (y - f(w, x))
    w_grad = -1*res[0]*x
    return w_grad


def run(x_train, y_train, function, error, alpha, epsilion=1E-9):

    global f
    f = function
    print(f"Starting Stocastic Gradient Descent with alpha={alpha}, epsilion={epsilion}\n")
    w       = np.ones(x_train.shape[1]+1)
    # w[0], w[1], w[2] = 605.14287044, 4.21484986, -10.92459524

    # x       = np.ones(x_train.shape[0]*(x_train.shape[1]+1)).reshape((x_train.shape[0], x_train.shape[1]+1))
    # insert column for x^0 or const factor x0 at starting of dataset (index = 0)
    x_train.insert(0, "Const", np.ones(x_train.shape[0]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    itr = 0
    prev_error = 1000000000

    while True:
        for itr in range(x_train.shape[0]):
            w = w - (alpha*grad_f(w, x_train[itr], y_train[itr]))
        
        new_error = error(w, x_train, y_train)
        if prev_error-new_error < epsilion:
            return w
        elif new_error>prev_error+epsilion:
            print("new_error greater than old error")
            return w
        prev_error = new_error
        print(f"MSE: {new_error}, \tRMSE: {new_error**0.5}, \tWeights: {w}\n")

    return w