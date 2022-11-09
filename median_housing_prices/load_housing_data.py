import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from zlib import crc32

from median_housing_prices import HOUSING_DATA_LOCAL_PATH


def load_housing_data(housing_path=HOUSING_DATA_LOCAL_PATH):
    '''

    :param housing_path:
    :return:
    '''
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def random_sampling_using_hashing(identifier, test_size):
    return crc32(np.int64(identifier)) & 0xffffffff < test_size * 2**32


if __name__ == '__main__':
    df_housing = load_housing_data()
    print('this is just for debugging purpose')
    df_housing.info()
    df_housing['ocean_proximity'].value_counts()
    df_housing.describe()

    # drawing histogram of all variable
    # df_housing.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # making a categorical variable out of the median income column
    df_housing['income_cat'] = pd.cut(df_housing['median_income'],
                                      bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                                      labels=[1, 2, 3, 4, 5])

    # df_housing['income_cat'].hist()

    # since median income has been converted to a categorical variable, we can use
    # stratified sampling
    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in splits.split(df_housing, df_housing['income_cat']):
        strat_train_set = df_housing.iloc[train_index, :]
        strat_test_set = df_housing.iloc[test_index, :]

    print(df_housing['income_cat'].value_counts(normalize=True))
    print(strat_train_set['income_cat'].value_counts(normalize=True))
    print(strat_test_set['income_cat'].value_counts(normalize=True))

    # random sampling using hashing
    # it can be seen that random sampling doesn't do as well as stratified sampling
    row_index = df_housing.reset_index()['index']
    is_test_set = row_index.apply(lambda x: random_sampling_using_hashing(x, 0.2))
    train_set = df_housing.loc[~is_test_set, :]
    test_set = df_housing.loc[is_test_set, :]
    print(train_set['income_cat'].value_counts(normalize=True))
    print(test_set['income_cat'].value_counts(normalize=True))

    for set_ in (strat_test_set, strat_train_set):
        set_.drop('income_cat', inplace=True, axis=1)

    # From hereon we will work only on the training set and set the test set aside
    housing = strat_train_set.copy()
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

    print('this is just for debugging')






