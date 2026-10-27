# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_file, parse_line
from src.template_extractor import normalize_msg

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Log Anomaly Dashboard")
st.title("Log Anomaly Detector ")

# --- Data Processing Function ---
def process_and_display_data(data_source, source_name="data"):
    try:
        st.sidebar.info(f"Loading and parsing {source_name}...")

        if isinstance(data_source, str): # File path for sample data
            parsed_logs = list(parse_file(data_source))
        else: # Uploaded file object
            data_source.seek(0)
            decoded_file = data_source.read().decode('utf-8').splitlines()
            parsed_logs = [parse_line(line) for line in decoded_file]

        if not parsed_logs:
            st.warning("No logs were parsed from the source.")
            return

        df = pd.DataFrame(parsed_logs)
        df['norm_msg'] = df['msg'].fillna('').apply(normalize_msg)
        st.sidebar.success(f"{source_name.capitalize()} parsed successfully!")

        rf_model = joblib.load('models/rf_anomaly_detector.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

        st.header("Log Data Overview")
        st.dataframe(df.head(100), use_container_width=True)

        with st.spinner("Analyzing logs..."):
            X_transformed = vectorizer.transform(df['norm_msg'])
            scores = rf_model.predict_proba(X_transformed)[:, 1]
            df['is_anomaly'] = (scores > 0.5).astype(int)
            df['anomaly_score'] = scores

        st.header("Anomaly Detection Results")
        st.dataframe(df[df['is_anomaly'] == 1], use_container_width=True)
        st.success("Analysis complete!")

    except FileNotFoundError as e:
        st.error(f"File Not Found: '{e.filename}'. Please ensure model/data files exist.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Sidebar and Main Logic ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a raw log file", type=['log', 'txt'])
use_sample = st.sidebar.button("Load and Analyze Sample Data", type="primary")

if use_sample:
    process_and_display_data('data/sample_logs.log', source_name="sample data")
elif uploaded_file is not None:
    process_and_display_data(uploaded_file, source_name="uploaded file")
else:
    st.info("Welcome! Please load data using the sidebar to begin analysis.")