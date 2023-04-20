import pandas as pd

df = pd.read_csv('./data/cat_in_the_dat_train.csv')
df.columns

df['ord_4'].fillna('NONE', inplace=True)
df['ord_4'].value_counts()

# when the number of times a particular category is present in the training data is below a certain threshold,
# we are mapping it as 'RARE'.
df.loc[
    df['ord_4'].value_counts()[df['ord_4']].values < 2000,
    'ord_4'
] = 'RARE'