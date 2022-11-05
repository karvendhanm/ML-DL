import os
import tarfile
import urllib

from median_housing_prices import *


def fetch_housing_data(housing_data_web_url=config.HOUSING_DATA_WEB_URL,
                       housing_data_local_path=config.HOUSING_DATA_LOCAL_PATH):
    '''

    :param housing_data_web_url:
    :param housing_data_local_path:
    :return:
    '''

    os.makedirs(housing_data_local_path, exist_ok=True)
    tgz_path = os.path.join(housing_data_local_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_data_web_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(housing_data_local_path)
    housing_tgz.close()


if __name__ == '__main__':
    fetch_housing_data()
