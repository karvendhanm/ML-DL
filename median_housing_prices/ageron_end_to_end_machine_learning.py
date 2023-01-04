import os
from pathlib import Path

import pandas as pd
import sklearn
import sys
import tarfile
import urllib.request

from packaging import version

assert sys.version_info >= (3, 7)
assert version.parse(sklearn.__version__) >= version.parse('1.0.1')


def load_housing_data():
    '''

    :return:
    '''
    tarball_path = Path('datasets/housing.tgz')
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/housing.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path='datasets')
    return pd.read_csv(Path("datasets/housing/housing.csv"))


if __name__ == '__main__':
    housing = load_housing_data()

    # taking a quick look at the data structure
    print(housing.head())
    print(housing.info())
    print(housing['ocean_proximity'].value_counts())
    print(housing.describe())


    print('')

