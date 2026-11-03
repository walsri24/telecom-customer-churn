# Telecom Customer Churn Prediction

## a. Problem Statement

Customer churn is an important business problem for telecommunications
companies. When customers discontinue a service, the company may lose
recurring revenue and incur additional customer-acquisition costs.

The objective of this project is to build and evaluate machine learning
classification models that predict whether a telecom customer will churn
(`Yes`) or remain with the company (`No`). Five classification
algorithms are compared using the same dataset and evaluation metrics. A
Streamlit application is also developed to allow users to upload test
data, select a trained model, and inspect its evaluation results.

-   **Target variable:** `Churn`
-   **Classes:** `Yes` (churn) and `No` (not churn)
-   **Task:** Binary classification

## b. Dataset Description

This project uses the **Telco Customer Churn** dataset published on
Kaggle:

[Telecom Customer Churn Dataset --
Kaggle](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn)

The dataset contains customer account information, subscribed services,
contract details, payment information, and the churn outcome.

### Dataset characteristics

-   **Total instances:** 7,021
-   **Input columns in the source dataset:** 20, including the customer
    identifier
-   **Target column:** `Churn`
-   **Training data:** 5,616 rows
-   **Test data:** 1,405 rows

The data was divided into training and test sets. The training set was
used for model fitting, cross-validation, and hyperparameter selection.
The held-out test set was reserved for final evaluation.

### Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `customerID`                        Unique customer identifier;
                                      excluded from model predictors

  `gender`                            Customer gender

  `SeniorCitizen`                     Indicates whether the customer is a
                                      senior citizen

  `Partner`                           Whether the customer has a partner

  `Dependents`                        Whether the customer has dependents

  `tenure`                            Number of months the customer has
                                      stayed with the company

  `PhoneService`                      Whether phone service is subscribed

  `MultipleLines`                     Whether multiple phone lines are
                                      subscribed

  `InternetService`                   Type of internet service

  `OnlineSecurity`                    Whether online security is
                                      subscribed

  `OnlineBackup`                      Whether online backup is subscribed

  `DeviceProtection`                  Whether device protection is
                                      subscribed

  `TechSupport`                       Whether technical support is
                                      subscribed

  `StreamingTV`                       Whether streaming TV is subscribed

  `StreamingMovies`                   Whether streaming movies are
                                      subscribed

  `Contract`                          Customer contract type

  `PaperlessBilling`                  Whether paperless billing is
                                      enabled

  `PaymentMethod`                     Customer payment method

  `MonthlyCharges`                    Monthly amount charged to the
                                      customer

  `TotalCharges`                      Total amount charged to the
                                      customer

  `Churn`                             Target label: `Yes` or `No`
  -----------------------------------------------------------------------

`customerID` is retained as an identifier in the source data but is not
used as a predictive feature. The target labels are encoded as `1` for
`Yes` and `0` for `No` during model development.

### Preprocessing

Preprocessing is performed within the model pipelines so that the same
transformations are applied consistently during training and prediction.

-   Numerical features: median imputation followed by standardization.
-   Categorical features: most-frequent-value imputation followed by
    one-hot encoding.
-   One-hot encoding uses `handle_unknown="ignore"` so unseen categories
    in uploaded data do not cause prediction errors.
-   Fitted preprocessing steps are saved together with the trained
    models.

## c. GitHub Repository Link

**GitHub Repository:** [Telecom Customer Churn](https://github.com/walsri24/telecom-customer-churn)

The repository should contain the complete source code, Jupyter
notebooks, `requirements.txt`, test CSV data, saved model artifacts, and
this README file, as required by the assignment.

**Live Streamlit Application:** \[Add your deployed Streamlit Community
Cloud URL here\]

## d. Models Used

Five classification algorithms were implemented using the same dataset:

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Gaussian Naive Bayes
5.  Support Vector Machine (SVM)

### Evaluation metrics

-   **Accuracy:** Proportion of all predictions that are correct.
-   **AUC:** Area under the ROC curve; measures how well the model
    separates the two classes across classification thresholds.
-   **Precision:** Proportion of customers predicted to churn who
    actually churned.
-   **Recall:** Proportion of actual churn customers correctly
    identified.
-   **F1 Score:** Harmonic mean of precision and recall.
-   **MCC:** Matthews Correlation Coefficient; a balanced measure of
    classification quality that considers all four confusion-matrix
    outcomes.

### Model comparison

The following table reports tuned model results on the held-out test
set. For SVM, predictions are generated from the predicted churn
probability using a threshold of 0.50, so its results are consistent
with the probability-based threshold evaluation described later.

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.7409   | 0.8397 | 0.5070    | 0.7796 | 0.6144 | 0.4549 |
| Decision Tree       | 0.7338   | 0.8266 | 0.4983    | 0.7715 | 0.6055 | 0.4411 |
| KNN                | 0.7815   | 0.8239 | 0.5953    | 0.5457 | 0.5694 | 0.4241 |
| Gaussian Naive Bayes| 0.6847   | 0.8067 | 0.4488    | 0.8360 | 0.5840 | 0.4114 |
| Support Vector Machine (SVM) | 0.7907 | 0.8245 | 0.6060 | 0.5995 | 0.6027 | 0.4607 |


*Values are rounded to four decimal places. The SVM row uses a
probability threshold of 0.50. The threshold-optimized SVM result is
reported separately below.*

### Hyperparameter tuning

Hyperparameter selection was performed using `GridSearchCV` with
stratified 5-fold cross-validation and F1 score as the selection metric.
The search was conducted using the training data only; the held-out test
data was not used to select hyperparameters.

| Model                  | Selected Hyperparameters |
|------------------------|------------------------|
| Logistic Regression    | `C=1`, `penalty='l1'`, `solver='liblinear'`, `class_weight='balanced'` |
| Decision Tree          | `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=5`, `min_samples_split=2`, `class_weight='balanced'` |
| KNN                   | `n_neighbors=21`, `weights='uniform'`, `p=1` |
| Gaussian Naive Bayes   | `var_smoothing=1e-12` |
| SVM                   | `kernel='rbf'`, `C=1`, `gamma='scale'`, `class_weight='balanced'` |


### Observations about model performance

| ML Model Name | Observation About Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest AUC (0.8397), indicating strong class-ranking ability. It also achieved churn recall of 0.7796 and an F1 score of 0.6144. Its precision of 0.5070 indicates that some customers predicted to churn did not actually churn. |
| **Decision Tree** | Achieved an F1 score of 0.6055 and recall of 0.7715. Its AUC was 0.8266. The selected maximum depth of 5 and minimum leaf size of 5 constrain model complexity. Its results were competitive, although below Logistic Regression and threshold-optimized SVM on F1. |
| **KNN** | Achieved the highest accuracy (0.7815) and precision (0.5953) in the five-model comparison. However, its recall was 0.5457, meaning it missed a larger proportion of actual churn customers than the other tuned models. |
| **Gaussian Naive Bayes** | Achieved the highest recall (0.8360), making it effective at identifying actual churn customers. However, its precision (0.4488) and accuracy (0.6847) were the lowest among the compared models, indicating a relatively high number of false-positive churn predictions. |
| **Support Vector Machine** | At a probability threshold of 0.50, SVM achieved the highest precision (0.6060), accuracy (0.7907), and MCC (0.4607) in the comparison table. Its recall at this threshold was 0.5995. Adjusting its classification threshold improves the precision-recall balance for the churn class. |

### Threshold optimization and final model selection
The classification threshold determines the probability above which a
customer is classified as likely to churn. The default threshold of 0.50
is not necessarily the best choice when the objective is to balance
churn precision and recall.

Threshold selection was performed using a separate validation split
taken from the training data. Candidate thresholds were evaluated using
F1 score, and the held-out test set was not used to choose the
threshold.

Validation results selected:

-   Logistic Regression: threshold approximately `0.50`, validation F1 =
    `0.6352`.
-   SVM: threshold approximately `0.40`, validation F1 = `0.6333`.

The tuned SVM was then refitted using the full training dataset and
evaluated on the held-out test set with the selected threshold of
`0.40`.

**Threshold-optimized SVM test results**

| Threshold | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------|----------|--------|-----------|--------|--------|--------|
| 0.40      | 0.7822   | 0.8245 | 0.5753    | 0.6774 | 0.6222 | 0.4737 |


Compared with the SVM probability-based result at threshold 0.50,
lowering the threshold to 0.40 increased churn recall from 0.5995 to
0.6774 and F1 from 0.6027 to 0.6222. Precision decreased from 0.6060 to
0.5753. This reflects the trade-off between identifying more churn
customers and generating more false-positive predictions.

**Overall winner for this dataset: Support Vector Machine (SVM) with a
probability threshold of 0.40.**

The threshold-optimized SVM is selected as the final model because it
achieved the highest test F1 score (0.6222) and MCC (0.4737) among the
consistently evaluated probability-threshold results. Logistic
Regression achieved a higher AUC (0.8397), so it remains a strong
alternative when ranking quality is the main priority. The final model
choice prioritizes the balance between churn precision and recall
represented by F1 and MCC.


## Project links

-   **GitHub repository:** [Telecom Customer Churn](https://github.com/walsri24/telecom-customer-churn)
-   **Streamlit Community Cloud app:** [Customer Churn -- Streamlit](https://telecom-customer-churn-k4tvaenptdwvz2nkwajkhc.streamlit.app/)
-   **Dataset source:** [Kaggle -- Telecom Customer)
    Churn](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn)
