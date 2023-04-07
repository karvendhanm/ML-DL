import numpy as np

from abhishek_thakur.util import get_precision, get_true_positives, get_false_positives
from sklearn import metrics

'''
micro precision in a multi-class classification is nothing but calculating the true positive and false positive for
each class and putting them together and then find the precision.
'''

def micro_precision(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    # calculate the number of classes
    num_classes = len(np.unique(y_true))

    tp_count, fp_count = 0, 0
    for _class in list(np.unique(y_true)):
        revised_y_true = [1 if elem == _class else 0 for elem in y_true]
        revised_y_pred = [1 if elem == _class else 0 for elem in y_pred]

        tp_count += get_true_positives(revised_y_true, revised_y_pred)
        fp_count += get_false_positives(revised_y_true, revised_y_pred)

    return tp_count/(tp_count + fp_count)


y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

_micro_precision = micro_precision(y_true, y_pred)
print(f'the micro precision in the implemented function is: {_macro_precision}')
print(f'the micro precision in the sklearn implemented function is: '
      f'{metrics.precision_score(y_true, y_pred, average="micro")}')