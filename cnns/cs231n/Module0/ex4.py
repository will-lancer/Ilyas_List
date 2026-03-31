import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x)**2 * np.cos(17*x)

# plt.plot(x, y)
# plt.show()

arr = np.random.randn(5, 5)
plt.imshow(arr)
plt.show()