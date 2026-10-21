import streamlit as st
import pandas as pd
from pathlib import Path
from src.visualizations import plot_distribution, plot_scatter, plot_correlation_heatmap

# pages/ is one level below the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"

st.title("Data Exploration")

if not DATA_PATH.exists():
    st.error("Dataset not found. Please ensure the CSV exists at the expected location.")
else:
    df = pd.read_csv(DATA_PATH)

    st.subheader("Dataset preview (first 10 rows)")
    st.dataframe(df.head(10))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Distribution of each numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    st.subheader("Feature Distributions")
    for col in numeric_cols:
        st.write(f"**{col}**")
        fig = plot_distribution(df, col)
        st.pyplot(fig)

    # Scatter of Hours Studied vs Performance Index
    if "Hours Studied" in df.columns and "Performance Index" in df.columns:
        st.subheader("Hours Studied vs Performance Index")
        fig = plot_scatter(df, "Hours Studied", "Performance Index")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap (numeric only — categorical excluded automatically)
    st.subheader("Correlation Heatmap")
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)
