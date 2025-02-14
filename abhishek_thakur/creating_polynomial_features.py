import numpy as np
import pandas as pd

from sklearn import preprocessing

df = pd.DataFrame(np.random.rand(100, 2), columns = [f"f_{i}" for i in range(1, 3)])

# creating 2 degree polynomial features.
pf = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# polynomial features
poly_feats = pf.fit_transform(df)

# number of columns in the dataframe
num_feats = poly_feats.shape[1]
df_transformed = pd.DataFrame(poly_feats, columns=[f"f_{i}" for i in range(1, num_feats + 1)])

# binning - converting numerical features to categorical
df['f_bin_10'] = pd.cut(df['f_1'], bins=10, labels=False)
df['f_bin_100'] = pd.cut(df['f_1'], bins=100, labels=False)

# variance of the feature
df.f_1.var()
df.f_2.var()
df.f_bin_10.var()
df.f_bin_100.var()

# log transformation of a variable
df.f_1.apply(lambda x: np.log(1 + x)).var()

