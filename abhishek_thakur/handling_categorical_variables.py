import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import preprocessing


df = pd.read_csv('./data/cat_in_the_dat_train.csv')
# sns.countplot(x=df['target'])
# plt.show()

df.columns
df['target'].value_counts(normalize=True)
print(df['ord_2'].unique())

_dict_temperature = {
    'Freezing': 0,
    'Cold': 1,
    'Warm': 2,
    'Hot': 3,
    'Boiling Hot': 4,
    'Lava Hot': 5
}

df['ord_2'].value_counts(normalize=True)

# manual mapping
# Scikit-learn's label encoder will do the same job, but can't handle missing/Nan values.
# df.loc[:, 'ord_2'] = df['ord_2'].map(_dict_temperature)
# df['ord_2'].value_counts(normalize=True)

# label encoder don't handle missing values
# so fill the missing values first
df['ord_2'].fillna('NONE', inplace=True)
df['ord_2'].value_counts(normalize=True)

# initializing label encoder
lbl_enc = preprocessing.LabelEncoder()
df.loc[:, 'ord_2'] = lbl_enc.fit_transform(df.ord_2.values)
df['ord_2'].value_counts(normalize=True)



