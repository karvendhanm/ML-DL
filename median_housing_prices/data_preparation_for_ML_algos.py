import warnings

warnings.filterwarnings("ignore")

import numpy as np
import os
import pandas as pd

from median_housing_prices import HOUSING_DATA_LOCAL_PATH
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


def load_housing_data(housing_path=HOUSING_DATA_LOCAL_PATH):
    '''

    :param housing_path:    :return:
    '''
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


if __name__ == '__main__':

    is_plot = 0

    df_housing = load_housing_data()
    df_housing['income_cat'] = pd.cut(df_housing['median_income'],
                                      bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                                      labels=[1, 2, 3, 4, 5])

    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in splits.split(df_housing, df_housing['income_cat']):
        strat_train_set = df_housing.iloc[train_index, :]
        strat_test_set = df_housing.iloc[test_index, :]

    for set_ in (strat_test_set, strat_train_set):
        set_.drop('income_cat', inplace=True, axis=1)

    # putting the test set aside and focusing only on training dataset.
    housing = strat_train_set.copy()
    housing['rooms_per_household'] = housing[['total_rooms', 'households']].apply(lambda x: x[0] / x[1], axis=1)
    housing['bedrooms_per_room'] = housing[['total_bedrooms', 'total_rooms']].apply(lambda x: x[0] / x[1], axis=1)
    housing['population_per_household'] = housing[['population', 'households']].apply(lambda x: x[0] / x[1], axis=1)

    corr_matrix = housing.corr()

    # prepare the data for machine learning algorithms.
    # dropping the target variable
    housing = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()

    # data cleaning
    imputer = SimpleImputer(strategy='median')
    # median can only be calculated on numerical attributes.
    # creating a copy of the data without non-numerical attributes
    housing_num = housing.drop('ocean_proximity', axis=1)
    imputer.fit(housing_num)

    housing_num.info()
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    housing_tr.info()

    # handling text and categorical attributes
    housing_cat = housing[['ocean_proximity']]
    housing_cat.head(10)
    housing_cat.value_counts()

    # The drawback of ordinal encoder is ML algortihms will assume that two nearby values are more
    # similar to each other than two distant values.
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    ordinal_encoder.categories_

    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot
    # csr matrix (compressed sparse row matrix) into normal numpy dense array
    housing_cat_1hot.toarray()
    cat_encoder.categories_

    # Custom Transformers
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

        def __init__(self, add_bedrooms_per_room=True):
            '''

            :param add_bedrooms_per_room:
            :return:
            '''
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            '''

            :param X:
            :param y:
            :return:
            '''
            return self

        def transform(self, X):
            '''

            :param X:
            :return:
            '''
            rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
            population_per_household = X[:, population_ix]/X[:, households_ix]

            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]


    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    # feature scaling & transformation pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']


    full_pipeline = ColumnTransformer(
        [
            ('num', num_pipeline, num_attribs),
            ('cat', OneHotEncoder(), cat_attribs)
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    # select and train the model
    # training using regression model
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)


    print('this is just for debugging')
