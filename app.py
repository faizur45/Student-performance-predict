import streamlit as st
from pathlib import Path
import os

# Set project root for relative imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

# Sidebar navigation
st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.sidebar.title("Navigation")
pages = {
    "Home": "pages/1_Home.py",
    "Data Exploration": "pages/2_Data_Analysis.py",
    "Predict Performance": "pages/3_Predict_Performance.py",
    "Model Evaluation": "pages/4_Model_Evaluation.py",
}
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Dynamically run the selected page
page_path = PROJECT_ROOT / pages[selection]
if page_path.exists():
    with open(page_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, page_path.name, "exec"), globals())
else:
    st.error(f"Page script not found: {page_path}")
