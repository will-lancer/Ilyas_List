import numpy as np

def makeArr():
    list1 = [1, 2, 3]
    list2 = [0] * 3
    list3 = [0] * 3
    for i in range(3):
        list2[i] = list1[i] + 3
        list3[i] = list2[i] + 3
    arr = np.array([list1, list2, list3])
    return arr

arr1 = makeArr()

# print(arr.shape)
print(arr1)

arr1 += 69

print(arr1)

arr2 = arr1 + 12

arr3 = arr1 + arr2

print(arr2)
print(arr3)

arr4 = arr2 @ arr3

print(arr4)

print(arr4.ndim)
print(arr4.size)



