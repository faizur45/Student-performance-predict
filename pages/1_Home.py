import streamlit as st
import pandas as pd
from pathlib import Path

# pages/ is one level below the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"

st.title("Student Performance Prediction – Home")

st.header("Project Overview")
st.markdown(
    """
    **Objective**: Predict a student's *Performance Index* based on study-related factors.

    | Feature | Description |
    |---|---|
    | Hours Studied | Daily study hours (1–9) |
    | Previous Scores | Score in previous exam (40–99) |
    | Extracurricular Activities | Yes / No |
    | Sleep Hours | Hours of sleep per night (4–9) |
    | Sample Question Papers Practiced | Practice papers completed (0–9) |

    **Target**: Performance Index (10–100)  
    **Primary model**: Linear Regression (scikit-learn)  
    **Comparison models**: Decision Tree · Random Forest
    """
)

st.subheader("Dataset Summary")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isnull().sum().sum()))
    st.dataframe(df.head())
else:
    st.warning("Dataset not found. Please ensure the CSV exists at the expected location.")

st.info("Use the sidebar to navigate to **Data Exploration**, **Predict Performance**, or **Model Evaluation** pages.")
