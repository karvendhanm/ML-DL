import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    """
    A custom class for greedy feature selection
    """

    def feature_selection(self, X, y):
        pass

    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments.
        """

        # select features, return scores and selected indices
        scores, features = self.feature_selection(X, y)


if __name__ == "__main__":

    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)

    #transform data by greedy feature selection
    GreedyFeatureSelection()(X, y)
