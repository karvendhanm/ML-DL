import os
import pandas as pd

from median_housing_prices import HOUSING_DATA_LOCAL_PATH

def load_housing_data(housing_path = HOUSING_DATA_LOCAL_PATH):
    '''

    :param housing_path:
    :return:
    '''
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

if __name__ == '__main__':
    df_housing = load_housing_data()
    print('this is just for debugging purpose')

