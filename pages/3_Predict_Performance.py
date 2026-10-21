import streamlit as st
import pandas as pd
from pathlib import Path
from src.predict import predict
from src.data_loader import load_data

# pages/ is one level below the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"

st.title("Predict Student Performance")

if not DATA_PATH.exists():
    st.error("Dataset not found. Please ensure the CSV exists at the expected location.")
else:
    df = load_data()

    target_col = "Performance Index"
    feature_cols = [c for c in df.columns if c != target_col]

    # Determine which columns are categorical vs numeric
    categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    st.subheader("Enter student details")
    input_data = {}

    with st.form(key="prediction_form"):
        for col in feature_cols:
            if col in categorical_cols:
                options = sorted(df[col].unique().tolist())
                input_data[col] = st.selectbox(col, options)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.slider(
                    col, min_value=min_val, max_value=max_val, value=mean_val
                )

        submit = st.form_submit_button("Predict Performance")

    if submit:
        try:
            pred = predict(input_data, model_name="LinearRegression")
            st.success(f"Predicted Performance Index: **{pred:.2f} / 100**")

            # Simple interpretation
            if pred >= 75:
                st.info("Excellent performance range. Keep up the great work!")
            elif pred >= 55:
                st.info("Good performance range. There is room for improvement.")
            else:
                st.warning("Below-average performance range. Consider increasing study hours or reviewing weak areas.")

            st.caption(
                "⚠️ This prediction is a statistical estimate from the trained Linear Regression model. "
                "It should not be taken as a definitive assessment of a student's abilities."
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
