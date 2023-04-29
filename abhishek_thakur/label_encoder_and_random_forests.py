import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv('./data/cat_in_the_dat_train_folds.csv')
    print(f'the columns in the dataframe: {df.columns}')

    features = [col for col in df.columns if col not in ['id', 'kfold', 'target']]
    print(f'the features selected are: {features}')

    # filling the missing values
    # convert all the columns to str. As all the columns are categorical variables
    # converting all the columns to str is alright.
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # label encoders
    # since we are going to use tree based algorithms, no need for
    # one hot encoding, label encoding itself is enough.
    lbl = preprocessing.LabelEncoder()
    for col in features:
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # getting training data
    df_train = df.loc[df['kfold'] != fold, :]

    # getting validation data
    df_valid = df.loc[df['kfold'] == fold, :]

    # getting training and validation data without any target and other unnecessary variables.
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # calculating area under the curve
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f'fold: {fold}, AUC: {auc} ')

    return None


if __name__ == '__main__':
    for fold in range(5):
        run(fold)

