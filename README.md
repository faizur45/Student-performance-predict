# Student Performance Prediction Using Machine Learning

> An interactive web application built with **Streamlit** that predicts a student's Performance Index based on study-related factors, using Linear Regression, Decision Tree, and Random Forest models.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

**Objective:** Predict a student's *Performance Index* (0–100) using five input features:

| Feature | Type | Range |
|---|---|---|
| Hours Studied | Numeric | 1 – 9 |
| Previous Scores | Numeric | 40 – 99 |
| Extracurricular Activities | Categorical | Yes / No |
| Sleep Hours | Numeric | 4 – 9 |
| Sample Question Papers Practiced | Numeric | 0 – 9 |

Three ML models are trained and compared:
- **Linear Regression** *(primary)*
- Decision Tree Regressor
- Random Forest Regressor

---

## Demo

Launch locally and open `http://localhost:8501` in your browser.

```bash
streamlit run app.py
```

### Pages
| Page | Description |
|---|---|
| 🏠 Home | Project overview, dataset summary, KPI cards |
| 📊 Data Exploration | Histograms, scatter plots, correlation heatmap |
| 🔮 Predict Performance | Interactive form → real-time prediction |
| 📈 Model Evaluation | Metric cards, Actual vs Predicted scatter, residual distribution, model comparison table |

---

## Dataset

The dataset (`data/Student_Performance.csv`) contains **10,000 rows** with 6 columns matching the schema above.

> **Note:** The dataset included in this repo is synthetically generated using NumPy to match the original schema. To use a real dataset, replace `data/Student_Performance.csv` and re-run `python run_train.py`.

---

## Architecture

```
Student_Performance.csv
        │
        ▼
 src/data_loader.py  ──► load_data()
        │
        ▼
src/preprocessing.py ──► preprocess()
   (OneHotEncoder + passthrough via ColumnTransformer)
        │
        ▼
 src/train_model.py  ──► train_models()
   ┌────────────────────────────────┐
   │  LinearRegression              │
   │  DecisionTreeRegressor         │
   │  RandomForestRegressor         │
   └────────────────────────────────┘
        │  saves .pkl to models/
        ▼
   src/predict.py   ──► predict(input_dict)
        │
        ▼
   Streamlit UI (app.py + pages/)
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/faizur45/Student-performance-predict.git
cd Student-performance-predict

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Train the models (creates models/*.pkl)
python run_train.py
```

---

## Usage

```bash
# Start the Streamlit app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Run Tests

```bash
pytest tests -v
```

Expected: **16 passed** ✅

---

## Evaluation Results

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Linear Regression** | 3.956 | 24.823 | 4.982 | **0.824** |
| Decision Tree | 5.711 | 53.045 | 7.283 | 0.624 |
| Random Forest | 4.418 | 31.259 | 5.591 | 0.779 |

> Linear Regression achieves the best R² (0.824) because the target variable has a near-linear relationship with the features by design.

---

## Project Structure

```
student-performance-prediction/
│
├── app.py                    # Streamlit entry point
├── run_train.py              # Training script
├── generate_dataset.py       # Synthetic dataset generator
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── Student_Performance.csv
│
├── models/
│   ├── LinearRegression.pkl
│   ├── DecisionTree.pkl
│   └── RandomForest.pkl
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Data_Analysis.py
│   ├── 3_Predict_Performance.py
│   └── 4_Model_Evaluation.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluation.py
│   └── visualizations.py
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_prediction.py
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Web UI |
| scikit-learn | ML models & preprocessing |
| pandas | Data manipulation |
| numpy | Numerical operations |
| plotly | Interactive charts |
| matplotlib / seaborn | Static plots |
| joblib | Model serialisation |
| statsmodels | OLS trendline in Plotly |
| pytest | Unit testing |

---

## Limitations & Future Work

**Limitations**
- The dataset is synthetic — real-world performance is influenced by many more factors.
- Linear Regression assumes a linear relationship; non-linear patterns are underfit.
- No hyperparameter tuning has been applied.

**Future Enhancements**
- [ ] Replace synthetic data with a real student performance dataset
- [ ] Add hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Add SHAP / LIME for model explainability
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Add user authentication and prediction history

---

## License

This project is for educational / internship purposes.
