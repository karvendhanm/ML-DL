import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')


def run(fold_):
    '''

    :param fold_:
    :return:
    '''

    df = pd.read_csv('./data/adult_folds.csv')

    # list of all numerical variables in the dataset
    numerical_columns = [
        'age',
        'fnlwgt',
        'education.num',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
    ]

    '''
    since we are going to use XGBoost, a tree based algorithm, we don't need one hot encoding
    as well as we can get away without normalization or standardization.
    '''

    features = [
                col
                for col in df.columns
                if col not in ['kfold',
                               'skfold',
                               'income',
                               'target']
                ]

    lbl = preprocessing.LabelEncoder()
    for col in features:
        # handle categorical variables.
        # there is no need to standardize or normalize the numerical data as we are using tree based algo.
        if col not in numerical_columns:

            # as we are processing only categorical variables,
            # converting all of them to string is alright.
            df.loc[:, col] = df[col].astype(str)
            df.loc[:, col] = df[col].str.replace('?', 'NONE')

            # label encoding on categorical variables
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    # splitting training and validation data based on kfolds.
    df_train = df.loc[df['kfold'] != fold_, :].reset_index(drop=True)
    df_valid = df.loc[df['kfold'] == fold_, :].reset_index(drop=True)

    model = xgb.XGBClassifier(n_jobs = -1)
    model.fit(df_train[features].values, df_train.target.values)
    valid_preds = model.predict_proba(df_valid[features].values)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {fold_}, auc: {auc}')

    return None


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)