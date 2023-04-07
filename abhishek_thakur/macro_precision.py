import numpy as np

from abhishek_thakur.util import get_precision
from sklearn import metrics
'''
macro precision in a multi-class classification is nothing but calculating the precision for each class seperately and
averaging them.
'''

def macro_precision(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    # calculate the number of classes
    num_classes = len(np.unique(y_true))

    precisions = []
    for _class in list(np.unique(y_true)):
        revised_y_true = [1 if elem == _class else 0 for elem in y_true]
        revised_y_pred = [1 if elem == _class else 0 for elem in y_pred]

        precisions.append(get_precision(revised_y_true, revised_y_pred))

    return np.sum(precisions)/num_classes


y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

_macro_precision = macro_precision(y_true, y_pred)
print(f'the macro precision in the implemented function is: {_macro_precision}')
print(f'the macro precision in the sklearn implemented function is: '
      f'{metrics.precision_score(y_true, y_pred, average="macro")}')