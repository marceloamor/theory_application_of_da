import numpy as np
from icecream import ic

a = np.arange(1, 26).reshape(5, 5)
ic(a[4, 2:4])

# print 4,5,9,10
# Method 1: Using direct indexing with array slicing
print("Method 1:", a[0:2, 3:5].flatten())

# Method 2: Using advanced indexing with lists
print("Method 2:", a[[0, 0, 1, 1], [3, 4, 3, 4]])

# Method 3: Using boolean masking
mask = np.zeros_like(a, dtype=bool)
mask[0, 3:5] = True
mask[1, 3:5] = True
print("Method 3:", a[mask])

# Method 4: Using np.take with flattened array
indices = [3, 4, 8, 9]  # These are the flattened indices for 4,5,9,10
print("Method 4:", np.take(a, indices))
