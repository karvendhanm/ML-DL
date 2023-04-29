import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

def run(_fold = 0):

    df = pd.read_csv('./data/adult_folds.csv')

    # Even as it looks liek there is no missing values,
    # All the missing values have been subsituted with '?'.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('?', 'NONE')

    # for col in df.columns:
    #     print(f'for feature: {col} the unique values are: {df[col].unique()}')

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

    # since all the columns here are categroical, it is okay to convert them into str datatype.
    for col in features:
        df_cat.loc[:, col] = df_cat[col].astype(str)

    # seperating out the train and the validation dataset
    df_train = df_cat.loc[df_cat['kfold'] != _fold, :].reset_index(drop=True)
    df_valid = df_cat.loc[df_cat['kfold'] == _fold, :].reset_index(drop=True)

    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {_fold}, auc: {auc}')

    return None


if __name__ == '__main__':
    for _fold in range(5):
        run(_fold)


