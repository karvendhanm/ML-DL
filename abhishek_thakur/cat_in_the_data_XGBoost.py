import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

def run(_fold):
    '''

    :param _fold: which of the stratified kfolds to be used as the holdout set.
    :return:
    '''

    df = pd.read_csv('./data/cat_in_the_dat_train_folds.csv')

    features = [col for col in df.columns if col not in ['id', 'target', 'kfold']]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # initialising label encoder
    lbl = preprocessing.LabelEncoder()

    for col in features:
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # training dataset
    df_train = df.loc[df['kfold'] != _fold, :]

    # validation dataset
    df_valid = df.loc[df['kfold'] == _fold, :]

    # getting training and validation data in an array format without target and other columns
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
                              n_jobs = -1,
                              max_depth=7,
                              n_estimators = 200
                              )

    model.fit(x_train, df_train.target.values)
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {_fold}, auc: {auc}')

    return None



if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)