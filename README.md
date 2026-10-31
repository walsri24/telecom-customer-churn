### Logistic Regression

Logistic Regression achieved an accuracy of **80.28%** and an AUC score of **0.8403**, indicating good overall classification and discriminatory performance. The model achieved a precision of 66.10%, recall of 52.42%, F1 score of 58.47%, and MCC of 0.4630 for the positive churn class.

The model performed better at identifying non-churning customers, achieving a recall of 90.32%, compared to 52.42% for churn customers. The confusion matrix shows that the model correctly identified 195 churn customers but failed to identify 177 actual churn customers.

Overall, Logistic Regression provides a reasonable baseline, but its ability to identify customers likely to churn needs improvement.

### Decision Tree Classifier

The Decision Tree Classifier achieved an accuracy of **71.46%** and an AUC score of **0.6498**. It obtained a precision of 46.42%, recall of 50.54%, F1 score of 48.39%, and MCC of 0.2877.

The confusion matrix shows that the model correctly identified 188 churn customers while incorrectly classifying 217 non-churning customers as churners. It also failed to identify 184 actual churn customers.

The tree selected `Contract_Month-to-month` as its root splitting feature, followed by features such as `MonthlyCharges` and `InternetService_Fiber optic`.

Compared with Logistic Regression, the Decision Tree demonstrated lower performance across all six evaluation metrics. This suggests that the unrestricted tree may not generalize as effectively to unseen customer records.

Overall, the baseline Decision Tree performed less effectively than Logistic Regression on this dataset.
