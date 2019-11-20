"""
This module implements standard gradient descent
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

f = None

# gradient function
def grad_f(w: np.array, x: np.array, y: np.array) -> int:
    res = (y - f(w, x))
    w_grad = -1*(res.transpose().dot(x)).flatten()

    return w_grad

# main runner for standard gradient descent algorithm
def run(x_train, y_train, function, error, alpha, epsilion=1E-9):

    global f
    f = function
    print(f"Starting Gradient Descent with alpha={alpha}, epsilion={epsilion}\n")
    w       = np.ones(x_train.shape[1]+1)
    x_train.insert(0, "Const", np.ones(x_train.shape[0]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    prev_error = 0
    prev_w = w
    itr_list = list()
    loss_list = list()
    n = 300
    flag = True
    while flag:
        for iterations in tqdm(range(300)):
            grad = grad_f(w, x_train, y_train) # alpha multiplied inside
            w = w - alpha*grad
            new_error = error(w, x_train, y_train)
            if new_error > 1e25:
                print("\nOVERSHOOT!!"*3)
                print(new_error)
                flag = False
                break
            if abs(new_error-prev_error) < epsilion:
                flag = False
                break
            prev_error = new_error

        itr_list.append(n)
        n += 300
        loss_list.append(new_error)
        print(f"Loss: {new_error}, \tWeights: {w}\n")
    
    plt.title(f'Gradient Descent')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.plot(itr_list, loss_list, label="loss function")
    plt.legend()
    plt.grid(True)
    plt.show()

    return w