import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

plot_points = np.random.random([2000, 1])

def plot(x,y,i):
    plt.figure(1)
    ax = plt.gca()
    mx = 12
    ax.set_ylim(0,mx)
    ax.xaxis.set_major_locator(mt.FixedLocator([i*0.1 for i in range(1,1)]))
    ax.plot(x,y,linewidth=0,marker='.',markersize=4)
    # plt.show()
    plt.savefig(f"Fig/{i}.png")
    plt.close()


def plot_beta_dist(a, b, i):
    x = plot_points
    const = gamma(a+b)/(gamma(a)*gamma(b))
    y = const*np.power(x,a-1)*np.power(1-x, b-1)
    plot(x, y, i)


def generate_tosses(number=160):
    np.random.seed(5554)    # fixed seed to get same data
    D = np.random.randint(2, size=number)
    D = 1-D     # to convert mean from 0.33 to 0.66
    while 0.4 <= D.mean() <= 0.6:
        D = np.random.randint(2, size=number)
    print(f"Mean of generated dataset: {D.mean()}")
    return D


def sequential_learning(dataset, a, b):
    print("Sequential learning")
    for i in range(len(dataset)):
        if dataset[i] == 1:  # i.e., toss is a head
            a += 1
        else:   # i.e., toss is a tail
            b += 1
        plot_beta_dist(a, b, i)


def complete_learning(dataset, a, b):
    print("Complete learning")
    m = np.sum(dataset)     # no of heads = no of ones
    l = len(dataset) - m    # no of tails = total - heads
    a += m
    b += l
    plot_beta_dist(a, b, 1000)


def main():
    dataset = generate_tosses()
    a,b = 4,6  # such that mean of B-dist is 0.4
    sequential_learning(dataset, a, b)  # Part A
    complete_learning(dataset, a, b)    # Part B


if __name__ == "__main__":
    print("Assignment 3")
    main()