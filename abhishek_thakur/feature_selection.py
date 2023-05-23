import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# create a random numpy array with 10 samples
# and 6 features and values ranging between 1 and 15.
X = np.random.randint(low=0, high=12, size=(10, 6))
df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(1, 7)])

for i in range(5):
    print(df.iloc[:, i].var())

scl = StandardScaler()
X_scl = scl.fit_transform(df)

for i in range(5):
    print(X_scl[:, i].var())

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(df)

for i in range(5):
    print(X_minmax[:, i].var())

var_thresh = VarianceThreshold(threshold=4)
transormed_data = var_thresh.fit_transform(df)