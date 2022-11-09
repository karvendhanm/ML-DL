import matplotlib.pyplot as plt
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

    df_housing['income_cat'].hist()

    # since median income has been converted to a categorical variable, we can use
    # stratified sampling
    split = StratifiedShuffleSplit()



    print('this is just for debugging')
