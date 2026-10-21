import pandas as pd
from pathlib import Path

# The data folder lives inside the project root (two levels up from this file).
# __file__ is src/data_loader.py → parents[0] = src, parents[1] = project root.
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Student_Performance.csv"

def load_data() -> pd.DataFrame:
    """Load the student‑performance dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all columns from the CSV.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)
