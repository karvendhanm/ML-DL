import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import tree
from sklearn import metrics

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

# reading the data
df_red_wine = pd.read_csv('./data/winequality-red.csv')

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

# splitting the independent and dependent variable of the data for model fitting.
X = df_train[idp_vars]
y = df_train[tgt_var]

train_accuracy, test_accuracy = [0.5], [0.5]
for _mx_depth in range(1, 25):

    # initializing decision tree classifier.
    tree_clf = tree.DecisionTreeClassifier(max_depth=_mx_depth)

    # fitting the decision tree model
    tree_clf.fit(X, y)

    training_data_prediction = tree_clf.predict(X)
    training_data_accuracy = metrics.accuracy_score(y, training_data_prediction)
    train_accuracy.append(round(training_data_accuracy, 3))

    testing_data_prediction = tree_clf.predict(df_test[idp_vars])
    testing_data_accuracy = metrics.accuracy_score(df_test[tgt_var], testing_data_prediction)
    test_accuracy.append(round(testing_data_accuracy, 3))


plt.figure(figsize=(10, 5))
sns.set_style('whitegrid')
plt.plot(train_accuracy, label='train accuracy')
plt.plot(test_accuracy, label='test_accuracy')
plt.legend(loc='upper left', prop={'size': 15})
plt.xticks(range(0, 26, 5))
plt.xlabel('max_depth', size=20)
plt.ylabel('accuracy', size=20)
plt.show()





