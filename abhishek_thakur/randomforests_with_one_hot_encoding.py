import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(_fold):
    '''

    :param _fold: which of the stratified kfolds to be used as the holdout set.
    :return:
    '''

    df = pd.read_csv('./data/cat_in_the_dat_train_folds.csv')

    # all columns except target, id and kfold
    features = [col for col in df.columns if col not in ['kfold', 'id', 'target']]

    # all the columns in this dataframe is of categorical data type
    # so it is okay to convert all of them to str type
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # training data with target
    df_train = df.loc[df['kfold'] != _fold, :]

    # validation data with target
    df_valid = df.loc[df['kfold'] == _fold, :]

    full_data = pd.concat([df_train[features],
                           df_valid[features]], axis=0)

    ohe = preprocessing.OneHotEncoder()
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    svd = decomposition.TruncatedSVD(n_components=120)

    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    rf = ensemble.RandomForestClassifier(n_jobs=-1)
    rf.fit(x_train, df_train.target.values)
    valid_preds = rf.predict_proba(x_valid)[:, 1]

    # get area under the curve(auc) score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {_fold}, auc: {auc}')

    return None


if __name__ == '__main__':
    for _fold in range(5):
        run(_fold)