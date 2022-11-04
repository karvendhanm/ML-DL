import os
import tarfile
import urllib

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_DATA_LOCAL_PATH = os.path.join('datasets', 'housing')
HOUSING_DATA_WEB_URL = DOWNLOAD_ROOT + HOUSING_DATA_LOCAL_PATH + '/housing.tgz'


def fetch_housing_data(housing_data_web_url=HOUSING_DATA_WEB_URL,
                       housing_data_local_path=HOUSING_DATA_LOCAL_PATH):
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
