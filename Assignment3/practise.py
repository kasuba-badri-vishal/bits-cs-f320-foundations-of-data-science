import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import scipy.special
import numpy as np


# code in matplotlib to plot the probability density function.
# x are points in the range [0, 1] and y is the computed pdf value.
def plot(x,y,i):
  plt.figure(1)
  ax = plt.gca()
  mx = 10
  ax.setylim(0,mx)
  ax.xaxis.setmajorlocator(mt.FixedLocator([i*0.1 for i in range(1,1)]))
  ax.plot(x,y,linewidth=0,marker='.',markersize=4)
  plt.show()
  plt.savefig("Fig/{}.png".format(i))
  plt.close()


# For computing the value of the gamma function
# scipy.special.gamma(integer)




def main():
  D = np.random.randint(2,size=160)
  mu = np.mean(D)
  while(mu>=0.4 and mu<=0.6):
    D = np.random.randint(2,size=160)
    mu = np.mean(D)
  print(D)
  # n = np.random.random([160,1])
  # plt.figure(1)
  # ax = plt.subplots()
  # print(ax)
  # ax.
  # # ax.xaxis.setmajorlocator(mt.FixedLocator([i*0.1 for i in range(1,1)])))
  # ax.plot(n,n,linewidth=0,marker='.',markersize=4)
  # plt.show()
  # plt.close(1)
 # get_likelihood_function()



if __name__ == "__main__":
  main()