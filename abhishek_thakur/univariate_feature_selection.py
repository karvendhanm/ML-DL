# Univariate feature selection using Mutual information, ANOVA F-test, and chi-squared.
# Univariate feature selection is nothing but scoring each feature against a given target.

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper built on
        difference univariate feature selection models from
        scikit-learn.
        :param n_features: Select Percentile if float else SelectKBest
        :problem problem_type: classification or regression
        :param scoring: scoring function, string
        """

        if problem_type == 'classification':
            valid_scoring = {
                'fclassif': f_classif,
                'chi2': chi2,
                'mutual_info_classif': mutual_info_classif
            }
        else:
            valid_scoring = {
                'f_regression': f_regression,
                'mutual_info_regression': mutual_info_regression
            }

        # raise exception if we do not have a valid scoring function
        if scoring not in valid_scoring:
            raise Exception('Invalid scoring function')

        # if n_features is int, we use selectkbest
        # if n_features is float, we use selectpercentile
        if isinstance(n_features, int):
            self.selection = SelectKBest(valid_scoring[scoring],
                                         k=n_features)
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception('Invalid type of feature')

    # fit function
    def fit(self, X, y):
        return self.selection.fit(X, y)

    # transform function
    def transform(self, X):
        return self.selection.transform(X)

    # fit_transform function
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)

# fetch a regression datatset
data = fetch_california_housing()

X = data['data']
y = data['target']

# convert to pandas dataframe
df = pd.DataFrame(X, columns=data['feature_names'])

# univariate feature selection
ufs = UnivariateFeatureSelection(
    n_features=0.1,
    problem_type='regression',
    scoring='f_regression'
)

ufs.fit(X, y)
X_transformed = ufs.transform(X)


