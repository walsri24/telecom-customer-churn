from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Models directory
MODEL_DIR = PROJECT_ROOT / "models"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"

# Test dataset
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

# Target column
TARGET_COLUMN = "Churn"

# Models available in the Streamlit application
MODEL_CONFIG = {
    "Baseline": {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    },
    "Hyperparameter Tuned": {
        "Logistic Regression": "logistic_regression_tuned.joblib",
        "Decision Tree": "decision_tree_tuned.joblib",
        "K-Nearest Neighbors": "knn_tuned.joblib",
        "Naive Bayes": "naive_bayes_tuned.joblib",
        "Support Vector Machine": "svm_tuned.joblib",
    },
    "Threshold Optimized": {
        "Logistic Regression": "logistic_regression_threshold_optimized.joblib",
        "Support Vector Machine": "svm_threshold_optimized.joblib",
    },
}

