"""Tests for src/predict.py — verify inference returns a valid float."""
import pytest
import math
from src.predict import predict, load_model

VALID_INPUT = {
    "Hours Studied": 6,
    "Previous Scores": 70,
    "Extracurricular Activities": "Yes",
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 4,
}


def test_predict_returns_float():
    result = predict(VALID_INPUT, model_name="LinearRegression")
    assert isinstance(result, float), "predict() must return a float"


def test_predict_finite():
    result = predict(VALID_INPUT, model_name="LinearRegression")
    assert math.isfinite(result), f"Prediction is not finite: {result}"


def test_predict_in_reasonable_range():
    result = predict(VALID_INPUT, model_name="LinearRegression")
    # Allow a small buffer beyond [10, 100] for edge predictions
    assert -10 <= result <= 110, f"Prediction {result} is wildly out of range"


def test_all_models_predict():
    for name in ("LinearRegression", "DecisionTree", "RandomForest"):
        result = predict(VALID_INPUT, model_name=name)
        assert math.isfinite(result), f"{name} prediction is not finite"


def test_load_model_returns_tuple():
    model, pipeline = load_model("LinearRegression")
    assert model is not None
    assert pipeline is not None


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        load_model("NonExistentModel")
