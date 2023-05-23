import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

# create a random numpy array with 10 samples
# and 6 features and values ranging between 1 and 15.
X = np.random.randint(low=0, high=2, size=(10, 6))
df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(1, 7)])

df.f_1.var()

var_thresh = VarianceThreshold(threshold=0.23)
transormed_data = var_thresh.fit_transform(df)