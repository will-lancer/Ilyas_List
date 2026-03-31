import numpy as np
from scipy.spatial.distance import cdist

X = np.random.randn(5, 3)
Y = np.random.randn(4, 3)

D = cdist(X, Y)


print(X)
print(Y)
print(D)