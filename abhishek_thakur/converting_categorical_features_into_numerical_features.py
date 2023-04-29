# converting categorical variables to numerical variables

import pandas as pd

df = pd.read_csv('./data/cat_in_the_dat_train.csv')
df.columns

# counting the number of times each categorical variable in the column 'ord_2' is repeated.
df.groupby('ord_2')['id'].count()

df['ord_2'].unique()
df.loc[:, 'ord_2'] = df['ord_2'].fillna('NONE')
df['ord_2'].unique()

df.loc[:, 'ord_2_transform'] = df.groupby('ord_2')['id'].transform('count')
df['ord_2_transform'].unique()

#it is even possible to combine 2 columns and use their counts
df['ord_1'].fillna('NONE', inplace=True)
df['ord_1'].unique()

df.columns

df.groupby(['ord_1', 'ord_2'])['id'].count().reset_index(name='count')


