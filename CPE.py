from sklearn import metrics
import pandas as pd

# clustering performance evaluation
labels_true = list(pd.read_excel('CLS.xlsx')['true'])
labels_pred = list(pd.read_excel('CLS.xlsx')['pred'])
print(metrics.rand_score(labels_true, labels_pred))

