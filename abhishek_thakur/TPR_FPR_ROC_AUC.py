import matplotlib.pyplot as plt

from abhishek_thakur.util import get_true_positive_rate, get_false_positive_rate
from sklearn import metrics

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

TPR_list = [] # TPR - true positive rate
FPR_list = [] # FPR - false positive rate

for thresh in thresholds:

    y_hat = [1 if proba >= thresh else 0 for proba in y_pred]
    tpr = get_true_positive_rate(y_true, y_hat)
    fpr = get_false_positive_rate(y_true, y_hat)

    TPR_list.append(tpr)
    FPR_list.append(fpr)

plt.figure(figsize=(7, 7))
plt.fill_between(FPR_list, TPR_list, alpha=0.4)
plt.plot(FPR_list, TPR_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()

# calculating the area under the curve.
_score = metrics.roc_auc_score(y_true, y_pred)
print(f'the area under the ROC curve is: {_score}')


