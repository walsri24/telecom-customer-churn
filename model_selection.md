# Final Model – Support Vector Machine (SVM)

## 1. Model Selection

After implementing and evaluating five baseline classification algorithms, hyperparameter tuning was performed using GridSearchCV with 5-Fold Stratified Cross-Validation.

The following classifiers were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

F1 Score was selected as the primary optimization metric because the telecom customer churn dataset contains an imbalanced target variable.

After comparing the baseline and tuned models, Support Vector Machine (SVM) was selected as the final model based on its overall F1 Score and Matthews Correlation Coefficient (MCC).

## 2. Hyperparameter Optimization

GridSearchCV was used to identify the optimal hyperparameter configuration for the SVM classifier.

### Best Hyperparameters

| Hyperparameter | Selected Value |
|---|---|
| C | 1 |
| Kernel | RBF |
| Gamma | scale |
| Class Weight | balanced |
| Probability | True |

### Hyperparameter Explanation

- **C = 1:** Controls the trade-off between maximizing the margin and minimizing classification errors.
- **Kernel = RBF:** Enables the model to learn nonlinear decision boundaries.
- **Gamma = scale:** Automatically determines the influence of individual training observations.
- **Class Weight = balanced:** Assigns greater importance to the minority Churn class.
- **Probability = True:** Enables probability estimation for classification threshold optimization.

The best configuration achieved a cross-validation F1 Score of **0.6240**.

## 3. Classification Threshold Optimization

By default, binary classification models commonly use a probability threshold of 0.5 to classify observations into positive and negative classes.

However, the default threshold may not provide the best balance between Precision and Recall, particularly when dealing with imbalanced datasets.

To improve churn detection, classification threshold optimization was performed on the tuned SVM model.

### Threshold Selection Procedure

1. The training dataset was divided into a subtraining dataset and a validation dataset using an 80:20 split.
2. The tuned SVM configuration was trained on the subtraining dataset.
3. Different classification thresholds were evaluated on the validation dataset.
4. F1 Score was used as the primary metric for selecting the optimal threshold.
5. The selected threshold was applied to the final model after refitting it on the complete training dataset.
6. The final model was evaluated using the held-out test dataset.

### Selected Classification Threshold

**Optimal threshold: 0.40**

The threshold was selected based on the highest validation F1 Score.

The probability-based classification rule is:

- If predicted churn probability is greater than or equal to 0.40, classify the customer as Churn.
- Otherwise, classify the customer as No Churn.

\[
\hat{y} =
\begin{cases}
1 & \text{if } P(\text{Churn}|X) \geq 0.40\\
0 & \text{otherwise}
\end{cases}
\]

Lowering the threshold makes the classifier more sensitive to potential churn customers, potentially increasing Recall while also increasing false positives.

## 4. Final Model Performance

The final SVM model was evaluated on the held-out test dataset containing 1,405 customer observations.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 0.7822 |
| AUC | 0.8245 |
| Precision | 0.5753 |
| Recall | 0.6774 |
| F1 Score | 0.6222 |
| Matthews Correlation Coefficient | 0.4737 |

### Interpretation

- **Accuracy (78.22%):** The model correctly classified approximately 78% of the test observations.
- **AUC (0.8245):** The model demonstrates good ability to distinguish between churn and non-churn customers.
- **Precision (57.53%):** Approximately 57.53% of customers predicted to churn actually churned.
- **Recall (67.74%):** The model identified approximately 67.74% of actual churn customers.
- **F1 Score (0.6222):** Provides a balance between Precision and Recall.
- **MCC (0.4737):** Indicates a positive correlation between predicted and actual classifications.

## 5. Baseline vs Tuned vs Threshold-Optimized SVM

The following table illustrates the effect of hyperparameter tuning and classification threshold optimization.

| Metric | Baseline | Tuned (0.5) | Threshold Optimized (0.4) |
|---|---:|---:|---:|
| Accuracy | 0.7979 | 0.7907 | 0.7822 |
| AUC | 0.7833 | 0.8245 | 0.8245 |
| Precision | 0.6618 | 0.6060 | 0.5753 |
| Recall | 0.4839 | 0.5995 | 0.6774 |
| F1 Score | 0.5590 | 0.6027 | 0.6222 |
| MCC | 0.4409 | 0.4607 | 0.4737 |

### Key Observations

1. Hyperparameter tuning improved the model's AUC from 0.7833 to 0.8245.
2. The tuned model improved Recall compared with the baseline configuration.
3. Threshold optimization increased Recall from 59.95% to 67.74%.
4. The threshold-optimized model achieved the highest F1 Score and MCC among these three probability-based SVM configurations.
5. The reduction in Precision indicates an increase in false-positive churn predictions.

Overall, threshold optimization improved the balance between Precision and Recall, making the model more suitable for identifying potential customer churn.

## 6. Final Model Configuration

The final model consists of:

- Support Vector Machine classifier.
- RBF kernel.
- Optimized hyperparameters obtained using GridSearchCV.
- Balanced class weights.
- Probability estimation enabled.
- Classification threshold set to 0.40.

The complete fitted preprocessing and classification pipeline is saved together with the selected threshold.

### Saved Model

```text
models/
└── final_churn_model.joblib
```

The saved artifact contains:

- Trained SVM pipeline.
- Classification threshold.
- Model name.
- Selected kernel.
- Final evaluation metrics.

## 7. Prediction Workflow

The final model follows the workflow below:

1. Receive customer information.
2. Apply numerical and categorical preprocessing.
3. Generate churn probability using the trained SVM.
4. Compare the probability against the selected threshold of 0.40.
5. Return the predicted customer churn status.

This allows the trained model to be integrated into a Python application for predicting customer churn using new customer information.

## 8. Conclusion

The Support Vector Machine classifier with an RBF kernel was selected as the final model after evaluating five classification algorithms and performing hyperparameter optimization.

Classification threshold optimization further improved its F1 Score from 0.6027 to 0.6222 and Recall from 0.5995 to 0.6774 compared with the tuned SVM using a probability threshold of 0.5.

The final model achieved an Accuracy of 78.22%, AUC of 82.45%, F1 Score of 62.22%, and MCC of 0.4737 on the held-out test dataset.

The experiment demonstrates that hyperparameter optimization and classification threshold selection can improve the effectiveness of machine learning models for imbalanced customer churn classification.