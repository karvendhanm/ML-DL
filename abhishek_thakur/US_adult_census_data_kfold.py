import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

from sklearn import model_selection

df = pd.read_csv('./data/adult.csv')

# we will do k-fold and stratified k-fold.
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

income_dict = {
    '<=50K':0,
    '>50K':1
}

df.loc[:, 'target'] = df['income'].map(income_dict)

# is the given dataset banlanced w.r.t its target column.
# since the target variable seems to be skewed, it would be better to
# use stratified k-fold.
# if sys.gettrace() is None:
#     sns.countplot(data=df, x='target')
#     plt.show()

df['target'].value_counts(normalize=True)

for _fold, (txn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = int(_fold)

for _kfold in range(5):
    df_kfold = df.loc[df['kfold'] == _kfold, :]
    print(df_kfold['target'].value_counts(normalize=True))

for _fold, (txn_, val_) in enumerate(skf.split(X=df, y=df['target'])):
    df.loc[val_, 'skfold'] = int(_fold)

for _skfold in range(5):
    df_skfold = df.loc[df['skfold'] == _skfold, :]
    print(df_skfold['target'].value_counts(normalize=True))

df.to_csv('./data/adult_folds.csv', index_label=False, index=False)

