"""Tests for src/preprocessing.py"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "Hours Studied": np.random.randint(1, 10, n),
        "Previous Scores": np.random.randint(40, 100, n),
        "Extracurricular Activities": np.random.choice(["Yes", "No"], n),
        "Sleep Hours": np.random.randint(4, 10, n),
        "Sample Question Papers Practiced": np.random.randint(0, 10, n),
        "Performance Index": np.random.uniform(10, 100, n),
    })


def test_preprocess_returns_five_objects(sample_df):
    result = preprocess(sample_df)
    assert len(result) == 5, "preprocess must return (pipeline, X_train, X_test, y_train, y_test)"


def test_split_sizes(sample_df):
    _, X_train, X_test, y_train, y_test = preprocess(sample_df, test_size=0.2)
    total = len(X_train) + len(X_test)
    assert total == len(sample_df.drop_duplicates()), "Train + test should equal total rows"
    assert abs(len(X_test) / total - 0.2) < 0.05, "Test fraction should be approximately 0.2"


def test_target_not_in_features(sample_df):
    _, X_train, X_test, y_train, y_test = preprocess(sample_df)
    assert "Performance Index" not in X_train.columns
    assert "Performance Index" not in X_test.columns


def test_pipeline_can_fit_and_transform(sample_df):
    pipeline, X_train, X_test, y_train, y_test = preprocess(sample_df)
    pipeline.fit(X_train)
    transformed = pipeline.transform(X_test)
    assert transformed.shape[0] == len(X_test)


def test_missing_target_raises(sample_df):
    df_bad = sample_df.drop(columns=["Performance Index"])
    with pytest.raises(KeyError):
        preprocess(df_bad)
