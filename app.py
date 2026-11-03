import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODEL_CONFIG
from src.model_loader import load_model
from src.evaluation import evaluate_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Telecom Customer Churn Prediction")

st.markdown("""
Evaluate classification models for predicting customer churn
using the Telecom Customer Churn dataset.
""")

st.divider()


# --------------------------------------------------
# MODEL CATEGORY SELECTION
# --------------------------------------------------
st.subheader("1. Select Model Category")

model_category = st.radio(
    "Choose an approach",
    options=["Baseline", "Hyperparameter Tuned", "Threshold Optimized"],
    horizontal=True
)


# --------------------------------------------------
# MODEL CATEGORY INFORMATION
# --------------------------------------------------
if model_category == "Baseline":

    with st.expander("About Baseline Models", expanded=True):
        st.markdown("""
        ### Baseline Classification Models

        This section evaluates five baseline machine learning
        classifiers trained using the original model configurations.

        The models are:

        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Support Vector Machine

        These models provide a baseline for comparing
        classification performance before hyperparameter tuning
        or classification threshold optimization.

        Evaluation metrics include Accuracy, AUC, Precision,
        Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
        """)

elif model_category == "Hyperparameter Tuned":

    with st.expander("About Hyperparameter Tuned Models", expanded=True):
        st.markdown("""
        ### Hyperparameter-Tuned Classification Models

        This section evaluates models after hyperparameter
        optimization using the training dataset.

        Hyperparameter tuning was performed to identify suitable
        model configurations and improve predictive performance.

        The tuned models include:

        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Support Vector Machine

        The saved tuned models are evaluated on the uploaded
        test dataset without retraining.

        Their performance can be compared against baseline models
        using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
        """)

elif model_category == "Threshold Optimized":

    with st.expander("About Threshold Optimized Models", expanded=True):
        st.markdown("""
        ### Threshold-Optimized Classification Models

        This section evaluates Logistic Regression and Support
        Vector Machine models with optimized classification
        decision thresholds.

        In binary classification, the default decision threshold
        is commonly 0.50. Threshold optimization changes the
        probability cutoff used to classify observations into
        positive and negative classes.

        This can alter the balance between Precision and Recall
        and may improve the F1 Score, depending on the selected
        threshold and the dataset.

        The threshold-optimized models were selected for further
        evaluation based on their classification performance.

        These saved models are evaluated directly on the uploaded
        test dataset without retraining.
        """)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
st.subheader("2. Select Classification Model")

available_models = list(MODEL_CONFIG[model_category].keys())

selected_model = st.selectbox(
    "Choose a model",
    options=available_models
)


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------
st.subheader("3. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your labelled test CSV file",
    type=["csv"]
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), width="stretch")

        if "Churn" not in df.columns:

            st.error(
                "Uploaded CSV must contain the Churn target column."
            )

        else:

            X_test = df.drop(
                columns=["Churn", "customerID"],
                errors="ignore"
            )

            y_test = df["Churn"]

            model = load_model(
                model_category,
                selected_model
            )

            if model_category == "Threshold Optimized":
                st.info(
                    f"Optimized classification threshold: "
                    f"{model.threshold:.4f}"
                )

            results = evaluate_model(
                model,
                X_test,
                y_test
            )

            metrics = results["metrics"]

            # ------------------------------------------
            # METRICS
            # ------------------------------------------
            st.divider()
            st.subheader("4. Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.4f}"
            )

            col2.metric(
                "AUC Score",
                f"{metrics['AUC']:.4f}"
                if metrics["AUC"] is not None
                else "N/A"
            )

            col3.metric(
                "Precision",
                f"{metrics['Precision']:.4f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{metrics['Recall']:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.4f}"
            )

            col6.metric(
                "MCC",
                f"{metrics['MCC']:.4f}"
            )

            # ------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------
            st.divider()
            st.subheader("5. Confusion Matrix")

            cm = results["confusion_matrix"]

            fig, ax = plt.subplots(figsize=(3, 2.5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                annot_kws={"size": 9},
                ax=ax,
                cbar=False
            )

            ax.set_xlabel("Predicted Label", fontsize=8)
            ax.set_ylabel("Actual Label", fontsize=8)
            ax.tick_params(axis="both", labelsize=8)

            plt.tight_layout()

            st.pyplot(fig, width="content")

            plt.close(fig)

            # ------------------------------------------
            # CLASSIFICATION REPORT
            # ------------------------------------------
            st.divider()
            st.subheader("6. Classification Report")

            report_df = pd.DataFrame(
                results["classification_report"]
            ).T

            st.dataframe(
                report_df.round(4),
                width="stretch"
            )

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info(
        "Upload a labelled test CSV to begin evaluation."
    )
