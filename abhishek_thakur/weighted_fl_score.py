import numpy as np

from abhishek_thakur.util import get_precision, get_recall
from collections import Counter
from sklearn import metrics


def weighted_f1_score(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    _class_dict = Counter(y_true)

    _f1 = 0
    for _class in list(np.unique(y_true)):
        revised_y_true = [1 if elem == _class else 0 for elem in y_true]
        revised_y_pred = [1 if elem == _class else 0 for elem in y_pred]

        _precision = get_precision(revised_y_true, revised_y_pred)
        _recall = get_recall(revised_y_true, revised_y_pred)

        if (_precision + _recall) != 0:
            temp_f1 = (2 * _precision * _recall) / (_precision + _recall)
        else:
            temp_f1 = 0

        _weighted_fl = temp_f1 * _class_dict[_class]

        _f1 += _weighted_fl

    return _f1/len(y_true)

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

_weighted_fl = weighted_f1_score(y_true, y_pred)
print(f'the weighted f1-score in the implemented function is: {_weighted_fl}')
print(f'the weighted f1-score in the sklearn implemented function is: '
      f'{metrics.f1_score(y_true, y_pred, average="weighted")}')