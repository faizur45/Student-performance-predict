import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

from src.evaluation import evaluate_model, plot_actual_vs_predicted, plot_residuals
# Correct module name is preprocessing (not preprocess)
from src.preprocessing import preprocess

# pages/ is one level below the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"

st.title("Model Evaluation Dashboard")

if not DATA_PATH.exists():
    st.error("Dataset not found. Please ensure the CSV exists at the expected location.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Reproduce the EXACT same train/test split used during training
# preprocess() returns an UNFITTED pipeline + the raw (DataFrame) splits
pipeline_unfitted, X_train, X_test, y_train, y_test = preprocess(
    df, test_size=0.2, random_state=42
)

model_names = ["LinearRegression", "DecisionTree", "RandomForest"]
metrics_dict = {}
preds_dict = {}

st.subheader("Evaluation Metrics")
cols = st.columns(len(model_names))

for i, name in enumerate(model_names):
    model_path = MODEL_DIR / f"{name}.pkl"
    if not model_path.exists():
        cols[i].error(f"{name} model not found. Run training first.")
        continue

    pkg = joblib.load(model_path)
    model = pkg["model"]
    # Each saved package contains the pipeline FITTED on training data
    fitted_pipeline = pkg["pipeline"]

    # X_test is still a raw DataFrame here — pass it directly to the fitted pipeline
    X_test_processed = fitted_pipeline.transform(X_test)
    y_pred = model.predict(X_test_processed)
    metrics = evaluate_model(y_test, y_pred)
    metrics_dict[name] = metrics
    preds_dict[name] = y_pred

    with cols[i]:
        st.markdown(f"**{name}**")
        st.metric("MAE", f"{metrics['MAE']:.3f}")
        st.metric("MSE", f"{metrics['MSE']:.3f}")
        st.metric("RMSE", f"{metrics['RMSE']:.3f}")
        st.metric("R²", f"{metrics['R2']:.3f}")

# Metric explanations
with st.expander("What do these metrics mean?"):
    st.markdown(
        """
        - **MAE** (Mean Absolute Error): Average absolute difference between actual and predicted values. Lower is better.
        - **MSE** (Mean Squared Error): Average squared difference. Penalises large errors more heavily. Lower is better.
        - **RMSE** (Root Mean Squared Error): Square root of MSE — in the same units as the target. Lower is better.
        - **R²** (R-squared): Proportion of variance explained by the model. 1.0 is perfect; 0.0 means the model is no better than the mean.
        """
    )

# Actual vs Predicted for Linear Regression
st.subheader("Actual vs Predicted (Linear Regression)")
if "LinearRegression" in preds_dict:
    fig = plot_actual_vs_predicted(y_test, preds_dict["LinearRegression"])
    st.plotly_chart(fig, use_container_width=True)

# Residual distribution for Linear Regression
st.subheader("Residuals Distribution (Linear Regression)")
if "LinearRegression" in preds_dict:
    fig_res = plot_residuals(y_test, preds_dict["LinearRegression"])
    st.plotly_chart(fig_res, use_container_width=True)

# Model comparison table
if metrics_dict:
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame(metrics_dict).T.round(4)
    st.dataframe(comparison_df)
