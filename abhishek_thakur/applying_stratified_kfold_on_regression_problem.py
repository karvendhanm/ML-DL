# Here we are going to use startified kfold on regression problem by binnig the target varaible.
# to determine the number of bins, we are going to use sturge's rule.

# stratified-kfold for regression
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    '''

    :param data:
    :return:
    '''
    data['kfold'] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    # to use kfold on a continuous data, we need to bin the continuous variable and create a new categorical column.
    # then use the new categorical column to create kfold.

    # let's calculate the ideal number of bins using sturge's law
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # cutting the numerical variable into bins
    data.loc[:, 'bins'] = pd.cut(data['target'], bins=num_bins, labels=False)

    # initializing Startified kfold class.
    kf = model_selection.StratifiedKFold(n_splits=5)

    for _fold, (t_, v_) in enumerate(kf.split(X=data, y=data.bins)):
        data.loc[v_, 'kfold'] = _fold

    # drop the bins column
    data.drop(labels=['bins'], axis=1, inplace=True)

    return data



if __name__ == '__main__':

    # we create a sample with 15000 samples, 100 independent variables and 1 target variable.
    X, y = datasets.make_regression(n_samples=15000, n_features=100, n_targets=1)

    # create a dataframe from the numpy arrays
    df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
    df.loc[:, 'target'] = y

    df = create_folds(df)




