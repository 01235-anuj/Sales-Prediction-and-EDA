import streamlit as st
import pandas as pd
import joblib

# ----------------- Page Config -----------------
st.set_page_config(page_title="🛒 Sales Prediction App", layout="centered")

st.title("🛍️ Sales Prediction App")
st.write("Enter details below and get sales predictions instantly.")

# ----------------- Load Model & Features -----------------
model = joblib.load("model_pipeline.pkl")
features_info = joblib.load("features_info.pkl")

# ----------------- Feature Lists -----------------
numeric_features = [
    'Unit price', 'Quantity', 'Tax 5%', 'Sales', 'cogs',
    'gross margin percentage', 'gross income', 'Rating',
    'day', 'month', 'year', 'weekday', 'total_price', 'profit_margin_pct'
]

categorical_features = [
    'Branch', 'City', 'Customer type', 'Product line', 'Payment'
]

# ----------------- User Input -----------------
st.subheader("📋 Input Data")

user_data = {}

# Numeric inputs
for feature in numeric_features:
    user_data[feature] = st.number_input(f"{feature}", value=0.0)

# Categorical inputs
for feature in categorical_features:
    user_data[feature] = st.text_input(f"{feature}")

# Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# ----------------- Prediction -----------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Value: {prediction}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
