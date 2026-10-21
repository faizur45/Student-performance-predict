import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

from .data_loader import load_data
from .preprocessing import preprocess

# src/ is one level below the project root, so parents[1] = project root
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_models(test_size: float = 0.2, random_state: int = 42):
    """Train Linear Regression (primary) and optional tree models.

    Returns a dict with fitted pipelines and evaluation metrics.
    """
    df = load_data()
    pipeline, X_train, X_test, y_train, y_test = preprocess(
        df, test_size=test_size, random_state=random_state
    )

    # Fit preprocessing on training data only, then transform both splits
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "pipeline": pipeline,
            "metrics": {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
            },
            "y_test": y_test,
            "y_pred": y_pred,
        }
        # Save model + fitted pipeline together so inference can reuse the same transformer
        joblib.dump({"model": model, "pipeline": pipeline}, MODEL_DIR / f"{name}.pkl")
        print(f"{name} metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    return results
