'''
log loss formula

log loss = -1 * (target * log(prediction)) + (1 -target) * log(1 - prediction)
'''

import numpy as np

from sklearn import metrics

def log_loss(y_true, y_proba):
    '''

    :param y_true:
    :param y_proba:
    :return:
    '''

    _loss = []
    epsilon = 1e-15
    for yt, yp in zip(y_true, y_proba):
        yp = np.clip(yp, epsilon, 1 - epsilon)
        _pred = np.log(yp)
        _neg_pred = np.log(1 - yp)

        res = -1 * ((yt * _pred) + ((1 - yt) * _neg_pred))
        _loss.append(res)

    return np.mean(_loss)


# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,
          0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
          0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
          0.85, 0.15, 0.99]

log_loss(y_true, y_pred)
metrics.log_loss(y_true, y_pred)

