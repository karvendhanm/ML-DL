import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from sklearn import tree
from sklearn import metrics

df_red_wine = pd.read_csv('./data/winequality-red.csv')

# there are only 6 different buckets in the quality column
df_red_wine['quality'].unique()  # array([5, 6, 7, 4, 8, 3])

quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}

df_red_wine.loc[:, 'quality'] = df_red_wine.quality.map(quality_mapping)
df_red_wine['quality'].unique()  # array([2, 3, 4, 1, 5, 0])

# independent variables
idp_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

# target variable
tgt_var = 'quality'

# shuffling the dataset and splitting the dataframe into train and test set.
df_red_wine = df_red_wine.sample(frac=1, random_state=42).reset_index(drop=True)

# first 1000 rows for training and rest of the rows for testing
df_train = df_red_wine.head(1000)
df_test = df_red_wine.tail(599)

# initializing decision tree classifier with a max_depth of 3.
tree_clf = tree.DecisionTreeClassifier(max_depth=3)

# splitting the independent and dependent variable of the data for model fitting.
X = df_train[idp_vars]
y = df_train[tgt_var]

# fitting the training data
tree_clf.fit(X, y)

# making predictions on the training data itself
training_data_predictions = tree_clf.predict(X)

# making predictions on the test data
test_data_predictions = tree_clf.predict(df_test[idp_vars])

# calculate the accuracy of predictions on training data set.
training_data_accuracy = metrics.accuracy_score(y, training_data_predictions)
print(f'the accuracy on training data set itself is: {training_data_accuracy}')

# calculate the accuracy of predictions on test data set.
test_data_accuracy = metrics.accuracy_score(df_test[tgt_var], test_data_predictions)
print(f'the accuracy on test data set is: {test_data_accuracy}')

# changing the max depth of the decision tree classifier to 7.
tree_clf_7 = tree.DecisionTreeClassifier(max_depth=7)

# fitting the training data
tree_clf_7.fit(X, y)

# making predictions on the training data itself
training_data_predictions_7 = tree_clf_7.predict(X)

# making predictions on the test data
test_data_predictions_7 = tree_clf_7.predict(df_test[idp_vars])

# calculate the accuracy of predictions on training data set.
training_data_accuracy_7 = metrics.accuracy_score(y, training_data_predictions_7)
print(f'the accuracy on training data set with max. depth 7 is: {training_data_accuracy_7}')

# calculate the accuracy of predictions on test data set.
test_data_accuracy_7 = metrics.accuracy_score(df_test[tgt_var], test_data_predictions_7)
print(f'the accuracy on test data set with max. depth 7 is: {test_data_accuracy_7}')




