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


### K-Nearest Neighbors (KNN)

The K-Nearest Neighbors classifier achieved an accuracy of **75.94%** and an AUC score of **0.7720**. It obtained a precision of 55.12%, recall of 49.19%, F1 score of 51.99%, and MCC of 0.3611.

The confusion matrix shows that the model correctly identified 183 churn customers while missing 189 actual churn customers. It also incorrectly classified 149 non-churning customers as churners.

The model performed better at identifying non-churning customers, achieving a recall of 85.58%, compared to 49.19% for churn customers.

An additional experiment with different K values showed that increasing K improved AUC and generally improved accuracy within the tested range. However, these exploratory results were obtained using the test dataset and were not used to replace the baseline model.

Overall, KNN performed better than the baseline Decision Tree but did not outperform Logistic Regression.


### Gaussian Naive Bayes

The Gaussian Naive Bayes classifier achieved an accuracy of **68.47%** and an AUC score of **0.8067**. It obtained a precision of 44.88%, recall of 83.60%, F1 score of 58.40%, and MCC of 0.4114.

The confusion matrix shows that the model correctly identified 311 churn customers while missing only 61 actual churn customers. However, it incorrectly classified 382 non-churning customers as churners.

The model achieved the highest churn recall among the four baseline classifiers evaluated so far. Its relatively low precision indicates that a substantial number of customers predicted to churn did not actually churn.

The probability distribution also showed a concentration of predictions near zero and one, suggesting that the model frequently produces extreme probability estimates.

Overall, Gaussian Naive Bayes provides strong churn detection capability but involves a trade-off between recall and precision. Its performance may be useful in scenarios where identifying potential churners is prioritized over minimizing false positives.


### Support Vector Machine (SVM)

The Support Vector Machine classifier with an RBF kernel achieved an accuracy of **79.79%** and an AUC score of **0.7833**. It obtained a precision of 66.18%, recall of 48.39%, F1 score of 55.90%, and MCC of 0.4409.

The confusion matrix shows that the model correctly identified 180 churn customers while missing 192 actual churn customers. It incorrectly classified 92 non-churning customers as churners.

The model achieved the highest churn precision among the five baseline classifiers, although its precision was only marginally higher than Logistic Regression. It also correctly identified 941 non-churning customers.

The trained SVM used an RBF kernel with C=1.0 and gamma='scale'. It identified 2,654 support vectors across the training observations.

Overall, SVM demonstrated competitive accuracy and precision but did not outperform Logistic Regression in overall predictive performance.