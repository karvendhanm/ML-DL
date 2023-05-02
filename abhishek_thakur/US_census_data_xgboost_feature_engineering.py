import itertools
import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')


def featuring_engineering(df, cat_cols, count=2):
    '''

    :param df:
    :param cat_cols:
    :param count:
    :return:
    '''
    combinations = list(itertools.combinations(cat_cols, count))
    for c1, c2 in combinations:
        col_name = c1 + "_" + c2
        df.loc[:, col_name] = df[[c1, c2]].apply(lambda x: str(x[0]) + '_' + str(x[1]), axis=1)

    return df

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
    since we are using XGBoost, a tree based algorithm, we don't need one hot encoding
    as well as we can get away without normalization or standardization.
    '''

    # making a list of all features, both categorical and numerical, that
    # will be used in the model as predictors/input variables.
    features = [
                col
                for col in df.columns
                if col not in ['kfold', 'skfold', 'income', 'target']
               ]

    # making a list of all categorical variables that will be used as input variables.
    cat_cols = [
                col
                for col in features
                if col not in numerical_columns
                ]

    for col in cat_cols:
        # handle categorical variables.
        # there is no need to standardize or normalize the numerical data as we are using tree based algo.
        # as we are processing only categorical variables,
        # converting all of them to string is alright.
        df.loc[:, col] = df[col].astype(str)
        df.loc[:, col] = df[col].str.replace('?', 'NONE')

    # building additional columns
    df = featuring_engineering(df, cat_cols, 2)
    new_features = [
        col
        for col in df.columns
        if col not in ['kfold', 'skfold', 'income', 'target']
    ]

    lbl = preprocessing.LabelEncoder()
    for col in new_features:

        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # splitting training and validation data based on kfolds.
    df_train = df.loc[df['kfold'] != fold_, :].reset_index(drop=True)
    df_valid = df.loc[df['kfold'] == fold_, :].reset_index(drop=True)

    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7)
    model.fit(df_train[new_features].values, df_train.target.values)
    valid_preds = model.predict_proba(df_valid[new_features].values)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {fold_}, auc: {auc}')

    return None







if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)