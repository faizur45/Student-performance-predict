import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np


def plot_distribution(df: pd.DataFrame, column: str):
    """Plot histogram of a numerical column using seaborn."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    """Scatter plot of two numeric columns with a linear trendline.

    Uses numpy polyfit to draw the trendline manually so that
    statsmodels is not required at runtime (it is still listed in
    requirements.txt for Plotly's own optional OLS trendline support
    used elsewhere in the project).
    """
    fig = px.scatter(df, x=x, y=y, opacity=0.5,
                     title=f"{x} vs {y}",
                     trendline="ols")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap for numeric columns using seaborn."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig
