import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    """
    A custom class for greedy feature selection
    """

    def evaluate_score(self, X, y):
        """
        this function evaluates model on data and returns
        Area under the curve(AUC).
        Note: We fit the data and calculate AUC on the same data.
        essentially we are overfitting.
        But this is also a way to achieve greedy selection as
        k-fold will take k times longer.

        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, -1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def feature_selection(self, X, y):
        """
        this funtion does the greedy selection
        : param X: data, numpy array
        : param y: targets, numpy array
        : return: (best scores, best features)
        """
        # initialize best score and good feature list
        good_features = []
        best_scores = []

        # calculate the number of features
        num_features = X.shape[1]

        # infinite loop
        while True:
            # initiate best feature and score for this loop
            this_feature = None
            best_score = 0

            # loop over all the features
            for feature in range(num_features):
                # if feature is already in good_features list
                # skip this for loop
                if feature in good_features:
                    continue

                # selected_features list below is the list of all the
                # good features till now and the current feature.
                selected_features = good_features + [feature]

                # remove all the other features from the data
                xtrain = X[:, selected_features]

                # calculate the score, in our case AUC
                score = self.evaluate_score(xtrain, y)

                if score > best_score:
                    this_feature = feature
                    best_score = score

            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            if len(best_scores) >= 2:
                if best_scores[-1] < best_scores[-2]:
                    break

        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments.
        """

        # select features, return scores and selected indices
        scores, features = self.feature_selection(X, y)

        # transform the data with selected features
        return X[:, features], scores


if __name__ == "__main__":

    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)

    #transform data by greedy feature selection
    X_transformed, scores = GreedyFeatureSelection()(X, y)
