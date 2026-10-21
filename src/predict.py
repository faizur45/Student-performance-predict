import pandas as pd
from pathlib import Path
import joblib
import numpy as np

# src/ is one level below the project root
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_model(model_name: str = "LinearRegression"):
    """Load a trained model and its fitted pipeline from disk.

    Parameters
    ----------
    model_name : str
        Filename stem (without .pkl extension).

    Returns
    -------
    (model, fitted_pipeline)
    """
    model_path = MODEL_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    obj = joblib.load(model_path)
    return obj["model"], obj["pipeline"]


def predict(input_dict: dict, model_name: str = "LinearRegression") -> float:
    """Return a single numeric prediction for the given feature dict.

    Parameters
    ----------
    input_dict : dict
        Keys must match the original feature column names
        (all columns except 'Performance Index').
    model_name : str
        Which saved model to use.

    Returns
    -------
    float
    """
    model, pipeline = load_model(model_name)
    # Build a one-row DataFrame so the pipeline receives the correct column names
    df = pd.DataFrame([input_dict])
    X_processed = pipeline.transform(df)
    pred = model.predict(X_processed)
    return float(pred[0])
