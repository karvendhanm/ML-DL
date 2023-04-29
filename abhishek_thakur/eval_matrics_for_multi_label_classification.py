def precision_at_k(y_true, y_pred, k):
    '''

    :param y_true:
    :param y_pred:
    :param k:
    :return:
    '''

    if k == 0:
        return 0

    # since we are interested on only top k vslues of the prediction
    y_pred = y_pred[:k]

    # convert predctions to set
    pred_set = set(y_pred)

    # convert ground truth to set
    true_set = set(y_true)

    # find common values
    common_values = pred_set.intersection(true_set)

    return len(common_values)/len(y_pred[:k])


def average_precision_at_k(y_true, y_pred, k):
    '''
    Average precision at k (AP@k) is calculated using precision at k. For instance if we require AP@3, then
    calculate P@1 (precision at 1), P@2, P@3 and then divide the sum by 3. AP@3 = (P@1 + P@2 + P@3)/3.

    :param y_true:
    :param y_pred:
    :param k:
    :return:
    '''

    _precision_at_k_values = []
    for i in range(1, k+1):
        _precision_at_k_values.append(precision_at_k(y_true, y_pred, i))

    if not len(_precision_at_k_values):
        return 0

    return sum(_precision_at_k_values)/len(_precision_at_k_values)


def mean_average_precision_at_k(y_true, y_pred, k):
    '''
    mean average precision at k (MAP@k) is calculated by calulating the average of average precision at k
    for all the samples

    :param y_true:
    :param y_pred:
    :param k:
    :return:
    '''

    mean_average_precision_values = []
    for i in range(len(y_true)):
        mean_average_precision_values.append(average_precision_at_k(y_true[i], y_pred[i], k))

    if not len(mean_average_precision_values):
        return 0

    return sum(mean_average_precision_values)/len(mean_average_precision_values)


y_true = [
    [1, 2, 3],
    [0, 2],
    [1],
    [2, 3],
    [1, 0],
    []
]

y_pred = [
    [0, 1, 2],
    [1],
    [0, 2, 3],
    [2, 3, 4, 0],
    [0, 1, 2],
    [0]
]

for i in range(len(y_true)):
    for j in range(1, 4):

        print(f'''
            y_true = {y_true[i]},
            y_pred = {y_pred[i]},
            AP@{j} = {average_precision_at_k(y_true[i], y_pred[i], j)}                
        ''')

mean_average_precision_at_k(y_true, y_pred, 1)
mean_average_precision_at_k(y_true, y_pred, 2)
mean_average_precision_at_k(y_true, y_pred, 3)
mean_average_precision_at_k(y_true, y_pred, 4)
2





