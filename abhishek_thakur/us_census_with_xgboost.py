import numpy as np
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

    # Even as it looks like there is no missing values,
    # All the missing values have been subsituted with '?'.
    # replacing '?' with 'NONE'.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('?', 'NONE')

    # in the first phase of the model, we will only focus on categorical variable
    # by dropping all the numerical variables
    numerical_columns = [
        'age',
        'fnlwgt',
        'education.num',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
    ]

    # dropping all the numerical columns
    df_cat = df.drop(labels=numerical_columns, axis=1)

    features = [col for col in df_cat.columns if col not in ['kfold', 'skfold', 'income', 'target']]

    '''
    since xgboost is a tree based algorithm, label encoder is enough.
    And, no need for one hot encoding. 
    '''

    # initalizing and label encoding all the categorical features
    lbl = preprocessing.LabelEncoder()
    for col in features:

        # since all the variables are of the type categorical, converting them to string doesn't
        # pose a threat.
        df_cat.loc[:, col] = df_cat[col].astype(str)
        lbl.fit(df_cat[col])
        df_cat.loc[:, col] = lbl.transform(df_cat[col])


    # splitting the training and testing data with target variables
    df_train = df_cat.loc[df_cat['skfold'] !=  fold_, :].reset_index(drop=True)
    df_valid = df_cat.loc[df_cat['skfold'] == fold_, :].reset_index(drop=True)

    model = xgb.XGBClassifier(n_jobs=-1)
    model.fit(df_train[features].values, df_train.target.values)
    valid_preds = model.predict_proba(df_valid[features].values)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {fold_}, auc: {auc}')

    return None


if __name__ == '__main__':
    for _fold in range(5):
        run(_fold)


