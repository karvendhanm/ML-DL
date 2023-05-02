# the biggest problem with target encoding is, it is too prone to overfitting.
# target encoding is used for categorical variables.

import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')


def mean_target_encoding(data):
    """

    :param data:
    :return:
    """

    # making a copy of the dataframe
    df = data.copy(deep = True)

    # list of all numerical variables in the dataset
    numerical_columns = [
        'age',
        'fnlwgt',
        'education.num',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
    ]

    # making a list of all features, both categorical and numerical, that
    # will be used in the model as predictors/input variables.
    # getting rid of few columns that are not actually inputs.
    features = [
        col
        for col in df.columns
        if col not in ['kfold', 'skfold', 'income', 'target']
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


    # a list to store 5 validation dataframes.
    encoded_dfs = []

    # go over all the folds
    for fold in range(5):

        # fetch training and validation data
        df_train = df.loc[df['kfold'] != fold, :].reset_index(drop = True)
        df_valid = df.loc[df['kfold'] == fold, :].reset_index(drop=True)

        for col in features:
            if col not in numerical_columns:
                mapping_dict = dict(df.groupby(col)['target'].mean())
                df_valid.loc[:, col + '_enc'] = df_valid[col].map(mapping_dict)
        encoded_dfs.append(df_valid)

    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df

def run(df, fold):
    """

    :param df:
    :param fold:
    :return:
    """

    df_train = df.loc[df['kfold'] != fold, :].reset_index(drop = True)
    df_valid = df.loc[df['kfold'] == fold, :].reset_index(drop = True)

    features = [
                col
                for col in df.columns
                if col not in ['kfold', 'skfold', 'income', 'target']
                ]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7)
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {fold}, auc: {auc}')


if __name__ == '__main__':

    # read data
    df = pd.read_csv('./data/adult_folds.csv')

    # create mean target encoded categories and munge data
    df = mean_target_encoding(df)

    for fold_ in range(5):
        run(df, fold_)

























































































































































































































































































































































































































































































































































































































