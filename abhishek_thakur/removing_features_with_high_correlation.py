# feature selection
# removing features with high correlation using pearson correlation coefficient.
import numpy as np
import pandas as pd

# getting califorina housing data
from sklearn.datasets import fetch_california_housing

# fetch a regression datatset
data = fetch_california_housing()

X = data['data']
y = data['target']

# convert to pandas dataframe
df = pd.DataFrame(X, columns=data['feature_names'])

# get pearson correlation coefficient matrix
df.corr()

df.loc[:, "MedInc_Sqrt"] = df["MedInc"].apply(np.sqrt)

# get pearson correlation coefficient matrix
df.corr()
