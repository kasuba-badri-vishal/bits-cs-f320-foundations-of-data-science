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
    loss_list = list()
    iters = list()
    flag=True
    while flag:
        for iterations in tqdm(range(500)):
            grad = grad_f(w, x_train, y_train) # alpha multiplied inside
            # prev_w = w
            w = w - alpha*grad
            # if w.dot(prev_w) > 0.01:
            #     print(w.dot(prev_w))
            new_error = error(w, x_train, y_train)
            loss_list.append(new_error)
            iters.append(iterations+1)
            if new_error > 1e20:
                print("\nOVERSHOOT!!"*3)
                print(new_error)
                flag=False
                break
            if abs(new_error-prev_error) < epsilion:
                flag=False
                break
            prev_error = new_error
            
        print(f"MSE: {new_error}, \tRMSE: {new_error**0.5}, \tWeights: {w}\n")
    
    plt.title(f'Loss/iteration - Linear regression')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.plot(iters, loss_list, label="Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return w