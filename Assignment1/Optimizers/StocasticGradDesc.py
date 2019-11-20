"""
This module implements stocastic gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt

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

    flag = True
    itr_list = list()
    loss_list = list()
    n = 1
    while flag:
        for itr in range(x_train.shape[0]):
            w = w - (alpha*grad_f(w, x_train[itr], y_train[itr]))
        new_error = error(w, x_train, y_train)
        itr_list.append(n)
        n += 1
        loss_list.append(new_error)
        if prev_error-new_error < epsilion or n == 16:
            flag = False
            break
        elif new_error>prev_error+epsilion:
            print("new_error greater than old error")
            flag = False
            break
        prev_error = new_error
        print(f"MSE: {new_error}, \tRMSE: {new_error**0.5}, \tWeights: {w}\n")
    
    plt.title(f'Loss/iteration - Linear regression')
    plt.ylabel('Loss')
    plt.xlabel('No of dataset iterations')
    plt.plot(itr_list, loss_list, label="Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return w