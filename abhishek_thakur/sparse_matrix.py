import numpy as np
from scipy import sparse

n_rows = 10000
n_cols = 100000

example = np.random.binomial(1, p=0.05, size=(n_rows, n_cols))

# print the size of the dense array, in bytes
print(f'the size of the dense array is: {example.nbytes}')

example_sparse = sparse.csr_matrix(example)

# print the size of the sparse array.
print(
    example_sparse.data.nbytes +
    example_sparse.indptr.nbytes +
    example_sparse.indices.nbytes
)

