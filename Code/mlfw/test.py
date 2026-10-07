import numpy as np
from sklearn import preprocessing

a = np.array([[100, 200, 300, 1000, 4000]])
b = np.array([[1, 2, 0.5, 0.25, 1.5]])

# ssdth = a*ssda + (1-a)ssdvb


a_norm = preprocessing.normalize(a, norm="l1")
b_norm = preprocessing.normalize(b, norm="l1")

print(a_norm)
print(np.sum(a_norm,axis=1))

print(b_norm)
print(np.sum(b_norm,axis=1))

alpha = 0.3

c = alpha*a_norm + (1-alpha)*b_norm

print(c)
print(np.sum(c,axis=1))
