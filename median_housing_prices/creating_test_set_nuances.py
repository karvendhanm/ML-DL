import numpy as np
import pandas as pd
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


if __name__ == '__main__':
    df_housing = load_housing_data()
    df_train, df_test = split_train_test(df_housing, 0.2)
    print('this is just for debugging')