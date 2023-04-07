import numpy as np

from abhishek_thakur.util import get_precision
from collections import Counter
from sklearn import metrics
'''
weighted precision in a multi-class classification is nothing but calculating the precision for each class seperately 
and assigning them weighteage as per the number of samples in the class. The bigger the number of samples in a class, 
the higher the weightage.
'''

def weighted_precision(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''

    _class_dict = dict(Counter(y_true))
    num_classes = len(_class_dict.keys())

    total = sum(_class_dict.values())
    final_dict = {_key : _class_dict[_key]/total for _key in _class_dict}

    _weigthed_precison = 0
    for _class in list(np.unique(y_true)):
        revised_y_true = [1 if elem == _class else 0 for elem in y_true]
        revised_y_pred = [1 if elem == _class else 0 for elem in y_pred]
        _weigthed_precison += get_precision(revised_y_true, revised_y_pred) * final_dict[_class]

    return _weigthed_precison


y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

_weighted_precision = weighted_precision(y_true, y_pred)
print(f'the weighted precision in the implemented function is: {_weighted_precision}')
print(f'the weighted precision in the sklearn implemented function is: '
      f'{metrics.precision_score(y_true, y_pred, average="weighted")}')
