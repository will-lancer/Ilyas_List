import numpy as np

# arr1 = np.arange(12)

# arr2 = arr1.reshape((3, 4))

# arr3 = arr2.reshape((4, 3))

# print(arr2)
# print(arr3)

# arr4 = np.arange(36)

# arr5 = arr4.reshape((3, 3, 4))
# arr6 = arr5.reshape((2, 3, 6))

# print(arr5)
# print(arr6)

row = np.arange(3)
col = np.arange(3)
col = col.reshape((3, 1))

arr = row * col

print(row)
print(col)
print(arr)