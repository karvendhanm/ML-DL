import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

if __name__ == "__main__":

    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)

    #transform data by greedy feature selection