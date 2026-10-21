import numpy as np, pandas as pd
from pathlib import Path

np.random.seed(42)
N = 10000
hours = np.random.randint(1, 10, size=N)
prev_scores = np.random.randint(40, 100, size=N)
extracurricular = np.random.choice(["Yes", "No"], size=N)
sleep_hours = np.random.randint(4, 10, size=N)
papers = np.random.randint(0, 10, size=N)
performance = (
    2.5 * hours +
    0.3 * prev_scores +
    1.5 * sleep_hours +
    2.0 * papers +
    np.where(extracurricular == "Yes", 5, 0) +
    np.random.normal(0, 5, size=N)
)
performance = np.clip(performance, 10, 100)

df = pd.DataFrame({
    "Hours Studied": hours,
    "Previous Scores": prev_scores,
    "Extracurricular Activities": extracurricular,
    "Sleep Hours": sleep_hours,
    "Sample Question Papers Practiced": papers,
    "Performance Index": performance.round(1)
})

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Student_Performance.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(DATA_PATH, index=False)
print(f"Generated dataset with {len(df)} rows at {DATA_PATH}")
