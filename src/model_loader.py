import joblib
import numpy as np
from src.config import MODEL_DIR, MODEL_CONFIG


class ThresholdOptimizedModel:

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        probabilities = self.model.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def load_model(model_category: str, model_name: str):

    if model_category not in MODEL_CONFIG:
        raise ValueError(f"Unknown model category: {model_category}")

    category_models = MODEL_CONFIG[model_category]

    if model_name not in category_models:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_DIR / category_models[model_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loaded_object = joblib.load(model_path)

    if model_category == "Threshold Optimized":

        model = loaded_object["model"]
        threshold = loaded_object["threshold"]

        return ThresholdOptimizedModel(
            model=model,
            threshold=threshold
        )

    return loaded_object
