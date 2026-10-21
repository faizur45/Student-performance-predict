import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true: pd.Series, y_pred: pd.Series):
    """Calculate regression metrics and return a dict."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series):
    """Scatter plot of actual vs predicted values with a reference line."""
    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(df, x="Actual", y="Predicted", trendline="ols",
                     title="Actual vs Predicted Performance Index")
    fig.add_shape(
        type="line",
        x0=y_true.min(), y0=y_true.min(),
        x1=y_true.max(), y1=y_true.max(),
        line=dict(dash="dash", color="gray"),
    )
    return fig

def plot_residuals(y_true: pd.Series, y_pred: pd.Series):
    """Residuals distribution plot."""
    residuals = y_true - y_pred
    fig = px.histogram(residuals, nbins=30, title="Residuals Distribution")
    return fig
