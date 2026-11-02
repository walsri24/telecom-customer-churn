import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODEL_CONFIG
from src.model_loader import load_model
from src.evaluation import evaluate_model


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Telecom Customer Churn Prediction")

st.markdown(
    """
    Evaluate machine learning classification models
    on telecom customer churn data.

    Upload a labelled test dataset, select a model,
    and view its performance metrics.
    """
)

st.divider()

# --------------------------------------------------
# Dataset upload
# --------------------------------------------------

st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"],
)

# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.subheader("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model",
    options=list(MODEL_CONFIG.keys()),
)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Validate target column
        if "Churn" not in df.columns:

            st.error(
                "Uploaded CSV must contain the Churn target column."
            )

        else:

            # Separate features and target
            X_test = df.drop(
                columns=["Churn", "customerID"],
                errors="ignore",
            )

            y_test = df["Churn"]

            # Load selected model
            model = load_model(selected_model)

            # Evaluate model
            results = evaluate_model(
                model,
                X_test,
                y_test,
            )

            metrics = results["metrics"]

            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            st.divider()
            st.subheader("3. Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.4f}",
            )

            col2.metric(
                "AUC Score",
                f"{metrics['AUC']:.4f}",
            )

            col3.metric(
                "Precision",
                f"{metrics['Precision']:.4f}",
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{metrics['Recall']:.4f}",
            )

            col5.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.4f}",
            )

            col6.metric(
                "MCC",
                f"{metrics['MCC']:.4f}",
            )

            # ------------------------------------------
            # Confusion Matrix
            # ------------------------------------------

            st.divider()

            st.subheader("4. Confusion Matrix")

            cm = results["confusion_matrix"]

            fig, ax = plt.subplots(figsize=(3, 3))

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
            # Classification Report
            # ------------------------------------------

            st.divider()
            st.subheader("5. Classification Report")

            report_df = pd.DataFrame(
                results["classification_report"]
            ).T

            st.dataframe(
                report_df.round(4),
                use_container_width=True,
            )

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info(
        "Upload a labelled test CSV to begin evaluation."
    )

