import numpy as np
import pandas as pd

from abhishek_thakur.util import get_true_positives, get_false_positives

# empty list to store true positive and false positive values
tp_list = []
fp_list = []

# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,
          0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
          0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
          0.85, 0.15, 0.99]

# different handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

# loop over all the thresholds
for thresh in thresholds:
    y_hat = [1 if _y >= thresh else 0 for _y in y_pred]

    tp = get_true_positives(y_true, y_hat)
    tp_list.append(tp)

    fp = get_false_positives(y_true, y_hat)
    fp_list.append(fp)

_arr = np.column_stack((np.array(thresholds).reshape(-1, 1), np.array(tp_list).reshape(-1, 1),
                np.array(fp_list).reshape(-1, 1)))

df = pd.DataFrame(_arr, columns=['threshold', 'true_positive', 'false_positve'])





