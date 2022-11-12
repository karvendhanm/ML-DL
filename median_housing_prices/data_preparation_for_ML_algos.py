import numpy as np
import os
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from median_housing_prices import HOUSING_DATA_LOCAL_PATH


def load_housing_data(housing_path=HOUSING_DATA_LOCAL_PATH):
    '''

    :param housing_path:
    :return:
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

    housing = strat_train_set.copy()
    housing['rooms_per_household'] = housing[['total_rooms', 'households']].apply(lambda x: x[0] / x[1], axis=1)
    housing['bedrooms_per_room'] = housing[['total_bedrooms', 'total_rooms']].apply(lambda x: x[0] / x[1], axis=1)
    housing['population_per_household'] = housing[['population', 'households']].apply(lambda x: x[0] / x[1], axis=1)

    corr_matrix = housing.corr()
