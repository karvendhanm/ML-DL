import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from abhishek_thakur.util import get_class, get_precision, get_recall, get_f1_score
from sklearn import metrics

# precision-recall curve.
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

y_pred = [0.02638412, 0.11114267, 0.31620708,
          0.0490937, 0.0191491, 0.17554844,
          0.15952202, 0.03819563, 0.11639273,
          0.079377, 0.08584789, 0.39095342,
          0.27259048, 0.03447096, 0.04644807,
          0.03543574, 0.18521942, 0.05934905,
          0.61977213, 0.33056815]

thresholds = [0.0490937, 0.05934905, 0.079377,
              0.08584789, 0.11114267, 0.11639273,
              0.15952202, 0.17554844, 0.18521942,
              0.27259048, 0.31620708, 0.33056815,
              0.39095342, 0.61977213]


precisions, recalls = [], []
for threshold in thresholds:
    y_hat = get_class(y_pred, threshold)
    P = get_precision(y_true, y_hat)
    R = get_recall(y_true, y_hat)

    precisions.append(P)
    recalls.append(R)

plt.figure(figsize=(7, 7))
plt.plot(recalls, precisions)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.show()

# _y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
#            1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#
# get_f1_score(y_true, _y_pred)
# metrics.f1_score(y_true, _y_pred)



