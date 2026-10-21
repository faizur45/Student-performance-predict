"""Tests for src/train_model.py — verify models train and pickles are created."""
import pytest
from pathlib import Path
from src.train_model import train_models

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
EXPECTED_MODELS = ["LinearRegression", "DecisionTree", "RandomForest"]


def test_train_models_returns_dict():
    results = train_models()
    assert isinstance(results, dict), "train_models() must return a dict"
    for name in EXPECTED_MODELS:
        assert name in results, f"'{name}' missing from training results"


def test_model_pickles_exist():
    """Pickle files must have been saved by train_models()."""
    for name in EXPECTED_MODELS:
        pkl_path = MODEL_DIR / f"{name}.pkl"
        assert pkl_path.exists(), f"{name}.pkl not found in models/"


def test_metrics_keys():
    results = train_models()
    for name, info in results.items():
        assert "metrics" in info, f"'metrics' key missing for {name}"
        for key in ("MAE", "MSE", "RMSE", "R2"):
            assert key in info["metrics"], f"'{key}' missing from {name} metrics"


def test_metrics_finite():
    import math
    results = train_models()
    for name, info in results.items():
        for key, val in info["metrics"].items():
            assert math.isfinite(val), f"{name} {key} is not finite: {val}"


def test_r2_positive():
    results = train_models()
    lr_r2 = results["LinearRegression"]["metrics"]["R2"]
    assert lr_r2 > 0, f"LinearRegression R² should be positive, got {lr_r2}"
