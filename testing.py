
import pandas as pd

from src.model_loader import load_model
from src.evaluation import evaluate_model

# Load test data
df = pd.read_csv("data/test_data.csv")

# Separate target and features
X_test = df.drop(columns=["Churn", "customerID"], errors="ignore")
y_test = df["Churn"]

# Load Logistic Regression
model = load_model("Support Vector Machine")

# Evaluate
results = evaluate_model(model, X_test, y_test)

# Print metrics
print("\nEvaluation Metrics")
print("----------------")

for metric, value in results["metrics"].items():
    print(f"{metric}: {value:.4f}")

print("\nConfusion Matrix")
print(results["confusion_matrix"])

print("\nClassification Report")
print(pd.DataFrame(results["classification_report"]).T)
