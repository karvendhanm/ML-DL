def get_class(probabilities, threshold=0.5):
    '''

    :param probabilities: list of probabilities predicted by the model for each sample.
    :param threshold: threshold at which positive or negative class is decided.
    :return: list of classes decided using the supplied threshold.
    '''

    _class = [int(proba >= threshold) for proba in probabilities]
    return _class

def get_true_positives(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    tp_counter = 0
    for sample_ground_truth, sample_prediction in zip(y_true, y_pred):
        if (sample_ground_truth == 1) & (sample_prediction == 1):
            tp_counter += 1

    return tp_counter

def get_false_positives(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    tp_counter = 0
    for sample_ground_truth, sample_prediction in zip(y_true, y_pred):
        if (sample_ground_truth == 0) & (sample_prediction == 1):
            tp_counter += 1

    return tp_counter

def get_false_negatives(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    tp_counter = 0
    for sample_ground_truth, sample_prediction in zip(y_true, y_pred):
        if (sample_ground_truth == 1) & (sample_prediction == 0):
            tp_counter += 1

    return tp_counter

def get_true_negatives(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    tp_counter = 0
    for sample_ground_truth, sample_prediction in zip(y_true, y_pred):
        if (sample_ground_truth == 0) & (sample_prediction == 0):
            tp_counter += 1

    return tp_counter

def get_precision(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    TP = get_true_positives(y_true, y_pred)
    FP = get_false_positives(y_true, y_pred)

    return TP/(TP + FP)

def get_recall(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    TP = get_true_positives(y_true, y_pred)
    FN = get_false_negatives(y_true, y_pred)

    return TP/(TP + FN)

def get_f1_score(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    P = get_precision(y_true, y_pred)
    R = get_recall(y_true, y_pred)

    return (2 * P * R)/(P + R)

def get_true_positive_rate(y_true, y_pred):
    '''
    TPR or true positive rate is also known as recall and sensitivity.
    :param y_true:
    :param y_pred:
    :return: we will just return recall function
    '''
    return get_recall(y_true, y_pred)

def get_true_negative_rate(y_true, y_pred):
    '''
    true negative rate or TNR is also known as specificity.
    :param y_true:
    :param y_pred:
    :return:
    '''
    FP = get_false_positives(y_true, y_pred)
    TN = get_true_negatives(y_true, y_pred)

    return TN/(TN + FP)

def get_false_positive_rate(y_true, y_pred):
    '''
    False positive rate or FPR is nothing but 1 - True negative rate with the formula FP/(TN + FP)
    :param y_true:
    :param y_pred:
    :return:
    '''

    return 1 - get_true_negative_rate(y_true, y_pred)

