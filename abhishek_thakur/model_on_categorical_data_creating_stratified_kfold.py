import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('./data/cat_in_the_dat_train.csv')

if sys.gettrace() is None:
    sns.countplot(x=df['target'])
    plt.show()

print(df['target'].value_counts(normalize=True))

# As the target variable in the train set is skewed,
# we will use stratified kfold

df['kfold'] = -1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for _fold, (_txn, _val) in enumerate(skf.split(X=df, y=df['target'])):
    df.loc[_val, 'kfold'] = _fold

df.to_csv('./data/cat_in_the_dat_train_folds.csv', index=False, index_label=False)