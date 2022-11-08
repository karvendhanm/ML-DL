import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zlib import crc32

from median_housing_prices import load_housing_data


# the following function even as it works will break when the data is updated.
def split_train_test(data, test_ratio):
    '''

    :param data:
    :param test_ratio:
    :return:
    '''
    np.random.seed(42)     # For reproducibility
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices, :], data.iloc[test_indices, :]


# using hashing to split the train and test set.
# hashing is immune to updating the dataset, and the train and test split remains the same.
def test_set_check(identifier, test_ratio):
    '''

    :param identifier:
    :param test_ratio:
    :return:
    '''
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    '''

    :param data:
    :param test_ratio:
    :param id_column:
    :return:
    '''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
    df_housing = load_housing_data()
    df_train, df_test = split_train_test(df_housing, 0.2)

    # using row index as unique identifier.
    housing_with_id = df_housing.reset_index()
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

    # this doesn't seem to be working well.
    housing_with_id['id'] = housing_with_id[['longitude', 'latitude']].apply(lambda x: (x[0]*1000) + x[1], axis=1)
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')

    # using scikit learn libraries
    train_set, test_set = train_test_split(df_housing, test_size=0.2, random_state=42)
