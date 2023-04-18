import numpy as np

from sklearn import preprocessing

example = np.random.randint(1000, size=1000000)

# initialize one hot encoder from scikit learn for dense matrix
ohe = preprocessing.OneHotEncoder(sparse=False)

ohe_example = ohe.fit_transform(example.reshape(-1, 1))
print(f'size of the one hot encoding example in the dense matrix form is: {ohe_example.nbytes}')

ohe_sparse = preprocessing.OneHotEncoder(sparse=True)
ohe_example_sparse = ohe_sparse.fit_transform(example.reshape(-1, 1))

print(f'size of the one hot encoding example in the sparse matrix form is: {ohe_example_sparse.data.nbytes}')

print(
    ohe_example_sparse.data.nbytes +
    ohe_example_sparse.indptr.nbytes +
    ohe_example_sparse.indices.nbytes
)
