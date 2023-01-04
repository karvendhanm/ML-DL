from packaging import version
from pathlib import Path

import pandas as pd
import sklearn
import sys
import tarfile
import urllib

# version check
assert sys.version_info >= (3, 7), 'the python version must be 3.7 or higher'
assert version.parse(sklearn.__version__) >= version.parse("1.0.1"), 'sklearn version must be 1.0.1 or higher'

def load_housing_data():
    '''

    :return:
    '''
    tarball_path = Path('datasets/housing.tgz')
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path='datasets')
    return pd.read_csv('datasets/housing/housing.csv')


housing = load_housing_data()


