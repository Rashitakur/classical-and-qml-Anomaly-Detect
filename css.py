import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set the page configuration at the very start
st.set_page_config(page_title="Anomaly Intruder Detection", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-color: #f0f0f5;  /* Light gray background */
            color: #333;  /* Dark text color */
        }
        .css-ffhzg2 {
            background-color: #2d2d72;  /* Dark blue sidebar */
            color: white;
        }
        .block-container {
            background-color: #ffffff;  /* White content background */
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #ff5c5c;  /* Red color for button */
            color: white;
            font-weight: bold;
        }
        h1, h2, h3 {
            color: #2d2d72;  /* Dark blue text color */
        }
    </style>
""", unsafe_allow_html=True)

# Load your trained Random Forest model
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\HP\DETECTION\DETECTION\rf_model1.pkl")  # Ensure the path is correct
    return model

model = load_model()

# App title
st.title("🔍 Anomaly Intruder Detection App")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predict", "📈 Visualization", "ℹ️ About"])

# Home page
if page == "🏠 Home":
    st.markdown("""
    ## 🚀 Welcome to the Anomaly Intruder Detection System
    This tool helps detect network intrusions using Machine Learning.
    """)
    
    
    st.markdown("""
    - Upload your network traffic data
    - Get anomaly predictions
    - Visualize patterns and explore results
    """)

# Predict page
elif page == "📊 Predict":
    st.header("📊 Upload Data for Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("🧾 Uploaded Data")
        st.dataframe(data.head())

        # Clean up the data (drop unwanted columns like 'Label' and 'Unnamed: 0')
        data_clean = data.drop(columns=["Label", "Unnamed: 0"], errors="ignore")

        # Ensure that the columns are aligned with the ones used during training
        model_columns = [col for col in data_clean.columns if col in model.feature_names_in_]
        X = data_clean[model_columns]

        # Make predictions
        try:
            prediction = model.predict(X)
            prediction_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            data_clean["Prediction"] = prediction
            if prediction_proba is not None:
                data_clean["Anomaly Score"] = prediction_proba

            st.success("✅ Prediction completed!")
            st.subheader("📋 Results")
            st.dataframe(data_clean.head())

            # Download option
            csv = data_clean.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "anomaly_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# Visualization page
elif page == "📈 Visualization":
    st.header("📈 Visualize Results")
    uploaded_file = st.file_uploader("Upload CSV with Predictions", type=["csv"], key="viz")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("👁️‍🗨️ Data Overview")
        st.dataframe(data.head())

        if "Prediction" in data.columns:
            count_plot = px.histogram(data, x="Prediction", title="Prediction Distribution", color="Prediction")
            st.plotly_chart(count_plot)

            if "Anomaly Score" in data.columns:
                st.subheader("📊 Anomaly Score Distribution")
                score_plot = px.histogram(data, x="Anomaly Score", nbins=50, title="Anomaly Score Histogram")
                st.plotly_chart(score_plot)
        else:
            st.warning("No 'Prediction' column found in the uploaded file.")

# About page
elif page == "ℹ️ About":
    st.header("ℹ️ About the Project")
    st.markdown("""
    This project is built using **Streamlit** and **Machine Learning/Quantum ML** for detecting intrusions or anomalies in network traffic data.

    - Developed by: **Rashita Thakur**
    - Tools: Python, Scikit-learn/machine,pandas, Streamlit, Pandas, Plotly
    - For: Academic/Research/Internship Project at CDAC

    Want to contribute or get the code? [GitHub Repo](#)
    """)

   