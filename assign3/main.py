import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import scipy.special as sc


def calculate_prior(D,mu_arr,a,b):
  coeff = sc.gamma(a+b)/(sc.gamma(a)*sc.gamma(b))
  print(coeff)
  arr1 = np.power(mu_arr,(a-1))
  arr2 = np.power((1-mu_arr),(b-1))
  beta_dis = coeff*(np.multiply(arr1,arr2))
  print(beta_dis)
  # plt.figure(1)
  # ax = plt.gca()
  # ax.plot(mu_arr,beta_dis,linewidth=0,marker='.',markersize=4)
  # plt.show()
  # plt.close(1)

def calculate_post(D,mu_arr,a,b):

  post_dis = np.zeros(160)

  for i in range(160):
    coeff = sc.gamma(a+b+1)/(sc.gamma(a+D[i])*sc.gamma(b+1-D[i]))
    temp1 = mu_arr[i]**(a-1+D[i])
    temp2 = (1-mu_arr[i])**(b-D[i])
    post_dis[i] = coeff*temp1*temp2
  plt.figure(1)
  ax = plt.gca()
  ax.plot(mu_arr,post_dis,linewidth=0,marker='.',markersize=4)
  plt.show()
  plt.close(1)

  print(post_dis)
  print(np.mean(post_dis))

def main():

  D = np.random.randint(2,size=160)
  mu = np.mean(D)
  while (mu>=0.4 and mu<=0.6):
    D = np.random.randint(2,size=160)
    mu = np.mean(D)
  a=0.075
  b=0.075
  mu_arr = np.random.random(160)

  coeff = sc.gamma(a+b)/(sc.gamma(a)*sc.gamma(b))
  # print(coeff)
  arr1 = np.power(mu_arr,(a-1))
  arr2 = np.power((1-mu_arr),(b-1))
  beta_dis = coeff*(np.multiply(arr1,arr2))
  print(np.mean(beta_dis))
  # calculate_post(D,mu_arr,a,b)


if __name__ == "__main__":
  main()