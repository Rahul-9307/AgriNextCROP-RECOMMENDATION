import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------
# SAFE IMAGE LOADING
# ---------------------------------------
def load_image(filename):
    return Image.open(os.path.join(os.path.dirname(__file__), filename))

banner = load_image("crop.png")
st.image(banner, use_column_width=True)

# ---------------------------------------
# SAFE CSV LOADING
# ---------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

# ---------------------------------------
# FEATURE SELECTION
# ---------------------------------------
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ---------------------------------------
# TRAIN MODEL
# ---------------------------------------
model = RandomForestClassifier(n_estimators=50, random_state=10)
model.fit(X, y)

# ---------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------
def predict_crop(n, p, k, temp, hum, ph, rain):
    data = np.array([[n, p, k, temp, hum, ph, rain]])
    prediction = model.predict(data)
    return prediction[0]


# ---------------------------------------
# MAIN UI FUNCTION
# ---------------------------------------
def main():

    # ---------------- HEADER BAR ----------------
    st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:12px 20px; background:#1e88e5; border-radius:8px;
                    margin-bottom:20px;">
            <h2 style="color:white; margin:0;">AgriNext</h2>
            <a href="https://rahul-9307.github.io/AgriNext/" target="_blank"
                style="padding:8px 16px; background:white; color:#1e88e5;
                       font-weight:600; border-radius:6px; text-decoration:none;">
                Home
            </a>
        </div>
    """, unsafe_allow_html=True)

    # ---------------- FLOATING BUTTON ----------------
    st.markdown("""
        <style>
        .floating-btn {
            position: fixed;
            bottom: 15px;
            right: 15px;
            background:#1e88e5;
            color:white;
            padding:12px 25px;
            border-radius:30px;
            font-size:15px;
            font-weight:bold;
            text-decoration:none;
            box-shadow:0 4px 10px rgba(0,0,0,0.3);
            z-index:9999;
        }
        </style>
        <a class="floating-btn" href="https://rahul-9307.github.io/AgriNext/" target="_blank">AgriNext</a>
    """, unsafe_allow_html=True)

    # ---------------- TITLE ----------------
    st.markdown("<h1 style='text-align:center;'>🌱 SMART CROP RECOMMENDATIONS</h1>",
                unsafe_allow_html=True)

    # ---------------- SIDEBAR INPUTS ----------------
    st.sidebar.title("AgriNext")
    st.sidebar.header("Enter Crop Details")

    nitrogen = st.sidebar.number_input("Nitrogen", 0.0, 140.0, 0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", 0.0, 145.0, 0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium", 0.0, 205.0, 0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 51.0, 0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0, step=0.1)
    ph_value = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0, step=0.1)

    # ---------------- PREDICT BUTTON ----------------
    if st.sidebar.button("Predict"):
        values = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall])

        if (values == 0).all():
            st.error("Please enter valid values before prediction.")
        else:
            crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall)
            st.success(f"🌾 Recommended Crop: **{crop}**")


# ---------------------------------------
# RUN APP
# ---------------------------------------
if __name__ == "__main__":
    main()
