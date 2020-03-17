import numpy as np
from scipy.stats import ortho_group

d = 10
n_train = 50000
n_test = 10000

def gen(n):
  z = np.random.normal(size=n)
  X = np.zeros([d, n])
  X[0] = z
  rot = ortho_group.rvs(d)
  X = rot.dot(X).T

Xtrain = gen(n_train)
np.save('d{}_n{}_std_train.npy'.format(d,n_train), Xtrain)
Xtest = gen(n_test)
np.save('d{}_n{}_std_test.npy'.format(d,n_test), Xtest)

