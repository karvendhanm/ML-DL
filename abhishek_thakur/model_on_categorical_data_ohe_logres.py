import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(_fold):
    '''

    :param _fold: which of the stratified kfolds to be used as the holdout set.
    :return:
    '''

    df = pd.read_csv('./data/cat_in_the_dat_train_folds.csv')

    features = [col for col in df.columns if col not in ['id', 'kfold', 'target']]

    # filling all the missing values
    # we are converting all the columns into a string datatype,
    # but it doesn't matter as they all are categorical variables.
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # creating the training and the validation sets
    df_train = df.loc[df['kfold'] != _fold, :].reset_index(drop=True)
    df_valid = df.loc[df['kfold'] == _fold, :].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )

    ohe.fit(full_data[features])

    # transform training data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize logistic regression model
    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {_fold}, AUC: {auc} ')

    return None

if __name__ == '__main__':
    for fold in range(5):
        run(fold)