import numpy as np
from sklearn import impute

# create a random numpy array with 10 samples
# and 6 features and values ranging between 1 and 15.
X = np.random.randint(low=1, high=15, size=(10, 6))

# convert the array to float
X = X.astype(float)

# randomly assign 10 elements to NaN (missing)
X.ravel()[np.random.choice(X.size, size=10, replace=False)] = np.nan

# use 2 nearest neighbors to fill na values
knn_imputer = impute.KNNImputer(n_neighbors=2)
X_imputed = knn_imputer.fit_transform(X)



