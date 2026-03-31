# NumPy / SciPy / Matplotlib Exercises (With Full Explanations)

This is a **ground-up, no-assumptions** set of exercises.  
I will **explain every concept and every piece of syntax** as it appears.  
You already know Python, so we focus only on *numerical computing ideas*.

---

## Core ideas you must understand first (before exercises)

### What is a NumPy array?
A NumPy array is an object of type:

```python
numpy.ndarray
````

It represents a **rectangular block of numbers in memory**.

Key properties of every NumPy array:

* **dtype** → what kind of numbers are stored
* **shape** → how many elements along each dimension
* **ndim** → number of dimensions (rank)

---

### What is `dtype`?

**`dtype` means “data type.”**

It tells you **what kind of numbers** the array stores:

* integers
* floating-point numbers
* booleans
* etc.

Examples:

```python
int64     # whole numbers
float64   # real numbers (decimals)
float32   # real numbers, less precision
bool      # True / False
```

Why this matters:

* All elements in a NumPy array have the **same dtype**
* Determines **precision**, **memory use**, and **speed**

Example:

```python
import numpy as np

a = np.array([1, 2, 3])
a.dtype        # int64

b = np.array([1.0, 2.0, 3.0])
b.dtype        # float64
```

---

### What is `shape`?

**`shape` tells you the size of the array along each axis.**

It is always a **tuple of integers**.

Examples:

```python
(5,)        # 1D array with 5 elements
(3, 4)      # 2D array: 3 rows, 4 columns
(10, 28, 28)# 3D array (e.g. images)
```

Example:

```python
A = np.zeros((3, 4))
A.shape      # (3, 4)
```

Think of shape as:

> “How many boxes in each direction?”

---

## Exercise 1 — Creating arrays and inspecting them

### Goal

Understand how NumPy arrays are created and how to inspect their structure.

### Syntax explained

```python
np.array([...])      # create array from Python list
np.zeros((m, n))     # array filled with zeros
np.ones((m, n))      # array filled with ones
np.eye(n)            # identity matrix
```

### Exercise

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.zeros((2, 3))
c = np.ones((4,))

print(a)
print(a.shape)
print(a.dtype)
```

**Explain to yourself:**

* Why is `a.shape == (3,)`?
* Why is `a.dtype` an integer type?

---

## Exercise 2 — Dimensions (`ndim`) and reshaping

### Goal

Understand what “dimension” means.

### Concepts

* **1D array** → vector
* **2D array** → matrix
* **3D array** → stack of matrices

### Syntax explained

```python
array.ndim           # number of dimensions
array.reshape(...)   # change shape without changing data
```

### Exercise

```python
x = np.arange(12)     # numbers 0–11
y = x.reshape((3, 4))

print(x.shape)        # (12,)
print(y.shape)        # (3, 4)
print(y.ndim)         # 2
```

**Important:**
Reshaping does **not** copy data — it reinterprets memory.

---

## Exercise 3 — Indexing and slicing (how to access data)

### Goal

Learn how to read and write array elements.

### Syntax explained

```python
A[i, j]       # element at row i, column j
A[i, :]       # entire row i
A[:, j]       # entire column j
A[a:b, c:d]   # submatrix
```

### Exercise

```python
A = np.arange(20).reshape((4, 5))

print(A[0, :])    # first row
print(A[:, 2])    # third column
print(A[1:3, 2:4])
```

**Key idea:**
NumPy slicing usually returns a **view**, not a copy.

---

## Exercise 4 — Vectorized arithmetic (no loops)

### Goal

Understand why NumPy is fast.

### Concept

Operations apply **element-by-element automatically**.

### Syntax explained

```python
A + B      # elementwise addition
A * B      # elementwise multiplication
A ** 2     # square each element
```

### Exercise

```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

z = x + y
print(z)
```

This replaces:

```python
for i in range(len(x)):
    z[i] = x[i] + y[i]
```

---

## Exercise 5 — Broadcasting (the most important idea)

### Goal

Understand how NumPy handles different shapes.

### Concept

Broadcasting stretches arrays **without copying data**.

### Rule (informal)

Dimensions are compatible if:

* they are equal, or
* one of them is 1

### Example

```python
A = np.ones((3, 1))
B = np.ones((1, 4))

C = A + B
C.shape     # (3, 4)
```

### Exercise

Manually reason:

* Why does `(3,1) + (1,4)` become `(3,4)`?

---

## Exercise 6 — Mathematical functions (ufuncs)

### Goal

Use NumPy’s built-in math.

### Syntax explained

```python
np.sin(x)
np.exp(x)
np.log(x)
np.maximum(x, 0)   # ReLU
```

### Exercise

```python
x = np.linspace(-2, 2, 100)
y = np.maximum(x, 0)
```

This is **exactly** how neural network activations are implemented.

---

## Exercise 7 — Linear algebra basics

### Goal

Learn matrix multiplication vs elementwise multiplication.

### Syntax explained

```python
A @ B        # matrix multiplication
A * B        # elementwise multiplication
```

### Exercise

```python
C = A @ B
print(C.shape)
```

Matrix multiplication rule:

```
(m × n) @ (n × p) → (m × p)
```

---

## Exercise 8 — SciPy distances

### Goal

Compute distances efficiently.

### Syntax explained

```python
from scipy.spatial.distance import cdist
```

### Exercise

```python
X = np.random.randn(5, 2)
Y = np.random.randn(3, 2)

D = cdist(X, Y)
print(D.shape)   # (5, 3)
```

Each entry is a Euclidean distance.

---

## Exercise 9 — Plotting with Matplotlib

### Goal

Visualize data.

### Syntax explained

```python
plt.plot(x, y)
plt.scatter(x, y)
plt.imshow(A)
plt.show()
```

### Exercise

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

---

## Exercise 10 — Images as arrays

### Goal

Understand images numerically.

### Concept

An RGB image has shape:

```
(height, width, 3)
```

### Exercise

```python
gray = image.mean(axis=2)
```

This averages color channels → grayscale.

---

## Final takeaway

By finishing these exercises, you will:

* Understand **what shape and dtype actually mean**
* Be able to reason about **array operations**
* Think in **vectorized linear algebra**
* Be fully prepared for CS231n assignments

This is the mental model CS231n assumes from day one.