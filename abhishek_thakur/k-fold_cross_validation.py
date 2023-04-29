import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import model_selection

quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}
df_wine_quality = pd.read_csv('./data/winequality-red.csv')
df_wine_quality.quality = df_wine_quality.quality.map(quality_mapping)

# df_wine_quality['quality'].hist()
# plt.show()

df_wine_quality = df_wine_quality.sample(frac=1).reset_index(drop=True)

# k-fold cross validation21
kf = model_selection.KFold(n_splits=5)
for _fold, (trn_, val_) in enumerate(kf.split(X=df_wine_quality)):
    df_wine_quality.loc[val_, 'kfold'] = int(_fold)

b = sns.countplot(x='quality', data=df_wine_quality)
b.set_xlabel('quality', fontsize=20)
b.set_ylabel('count', fontsize=20)

# startified k-fold cross validation
kf_strat = model_selection.StratifiedKFold(n_splits=5)
for _fold, (trn_, val_) in enumerate(kf_strat.split(X=df_wine_quality, y=df_wine_quality.quality)):
    df_wine_quality.loc[val_, 'kfold_strat'] = int(_fold)


