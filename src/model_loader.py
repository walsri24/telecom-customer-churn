import joblib

from src.config import MODEL_DIR, MODEL_CONFIG


def load_model(model_name: str):
    """
    Load a saved classification model.

    Parameters
    ----------
    model_name : str
        Name of the model selected in the application.

    Returns
    -------
    model
        Loaded sklearn model or pipeline.
    """

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_DIR / MODEL_CONFIG[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model

