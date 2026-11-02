import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def prepare_target(y):
    """
    Convert churn labels into binary values:
    No -> 0
    Yes -> 1
    """

    y = pd.Series(y)

    # Handle string labels, including pandas StringDtype
    if pd.api.types.is_string_dtype(y.dtype):
        y = y.astype(str).str.strip().map({
            "No": 0,
            "Yes": 1,
            "0": 0,
            "1": 1
        })

    # Validate conversion
    if y.isna().any():
        raise ValueError(
            "Invalid target labels found. "
            "Expected labels: 'No', 'Yes', 0, or 1."
        )

    return y.astype(int)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.

    Returns
    -------
    dict
        Evaluation metrics, confusion matrix, and classification report.
    """

    y_test = prepare_target(pd.Series(y_test))

    # Generate predictions
    y_pred = model.predict(X_test)

    # Convert predicted labels if necessary
    y_pred = prepare_target(pd.Series(y_pred))

    # Obtain probability for positive class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    # Evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_prob": y_prob if hasattr(model, "predict_proba") else None,
    }

