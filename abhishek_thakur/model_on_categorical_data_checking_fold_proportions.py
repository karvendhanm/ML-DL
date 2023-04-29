import pandas as pd

df = pd.read_csv('./data/cat_in_the_dat_train_folds.csv')

df.kfold.value_counts()
for _fold in list(df.kfold.unique()):
    df_tmp = df.loc[df['kfold'] == _fold, :]
    print(df_tmp['target'].value_counts())
