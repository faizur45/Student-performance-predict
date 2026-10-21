import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Preprocess the dataset and return train-test splits.

    Steps:
    - Drop duplicate rows
    - Separate features and target (Performance Index)
    - Identify categorical columns (e.g., Extracurricular Activities)
    - Apply One-Hot encoding to categorical features via a scikit-learn Pipeline
    - Split data into train and test using the provided random_state

    Returns
    -------
    (pipeline, X_train, X_test, y_train, y_test)
        pipeline is unfitted; call pipeline.fit_transform(X_train) inside the
        training module to avoid data leakage.
    """
    df = df.drop_duplicates().reset_index(drop=True)

    target_col = "Performance Index"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Use "str" (pandas 3 compatible) instead of "object" to silence deprecation warning
    # but keep "object" as fallback so the code works on pandas 2 too.
    try:
        categorical_cols = X.select_dtypes(include=["str", "category"]).columns.tolist()
        # pandas 2 with "str" returns nothing; fall back if empty
        if not categorical_cols:
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    except Exception:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return pipeline, X_train, X_test, y_train, y_test
