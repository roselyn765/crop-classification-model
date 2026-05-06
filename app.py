import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD TRAINED MODEL
# =========================
with open("crop_model.pkl", "rb") as file:
    model = pickle.load(file)

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🌱 Smart Crop Recommendation System")

st.markdown("""
This system helps farmers and users identify the **best crop to plant**
based on soil nutrients and environmental conditions.

### 📌 Required Inputs
- **Nitrogen (N)** → Helps plant leaf growth
- **Phosphorus (P)** → Helps root and flower development
- **Potassium (K)** → Helps overall plant health
- **Temperature** → Current environmental temperature
- **Humidity** → Moisture level in the air
- **pH Level** → Acidity or alkalinity of the soil
- **Rainfall** → Expected rainfall amount
""")

# =========================
# INPUT SECTION
# =========================
st.header("📋 Enter Soil and Weather Information")

# Soil Nutrients
st.subheader("🧪 Soil Nutrients")

N = st.number_input(
    "Nitrogen (N)",
    min_value=0,
    max_value=150,
    value=50,
    help="Recommended range: 0 - 140"
)

P = st.number_input(
    "Phosphorus (P)",
    min_value=0,
    max_value=150,
    value=50,
    help="Recommended range: 5 - 145"
)

K = st.number_input(
    "Potassium (K)",
    min_value=0,
    max_value=250,
    value=50,
    help="Recommended range: 5 - 205"
)

# Weather Conditions
st.subheader("🌦 Weather Conditions")

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    format="%.2f",
    help="Example: 20°C - 35°C"
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    format="%.2f",
    help="Percentage of moisture in the air"
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    format="%.2f",
    help="Expected rainfall amount in millimeters"
)

# Soil pH
st.subheader("🌱 Soil Condition")

ph = st.number_input(
    "Soil pH Level",
    min_value=0.0,
    max_value=14.0,
    value=6.5,
    format="%.2f",
    help="Most crops grow well between pH 5.5 and 7.5"
)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🌾 Predict Best Crop"):

    input_data = pd.DataFrame([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]], columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ])

    prediction = model.predict(input_data)

    st.success(f"✅ Recommended Crop to Plant: {prediction[0]}")

    st.info("""
    📌 Recommendation generated using a Machine Learning Random Forest Classifier model.
    """)