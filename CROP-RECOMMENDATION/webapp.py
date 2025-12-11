import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from PIL import Image

warnings.filterwarnings('ignore')

# ------------------------------
# SAFE IMAGE LOADING
# ------------------------------
def load_image(filename):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    return Image.open(file_path)

img = load_image("crop.png")
st.image(img, use_column_width=True)

# ------------------------------
# SAFE CSV LOADING
# ------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

# ------------------------------
# FEATURE SELECTION
# ------------------------------
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ------------------------------
# TRAIN MODEL
# ------------------------------
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X, y)

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_crop(n, p, k, temp, hum, ph, rain):
    data = np.array([[n, p, k, temp, hum, ph, rain]])
    prediction = model.predict(data)
    return prediction[0]


# ------------------------------
# STREAMLIT USER INTERFACE
# ------------------------------
def main():

    # --------------------------
    # CUSTOM HEADER WITH HOME BUTTON
    # --------------------------
    st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:10px 20px; background:#1e88e5; border-radius:8px;
                    margin-bottom:20px;">
            
            <h2 style="color:white; margin:0;">AgriNext</h2>

            <a href="https://rahul-9307.github.io/AgriNext/" target="_blank" style="
                padding:8px 16px;
                background:white;
                color:#1e88e5;
                border-radius:6px;
                text-decoration:none;
                font-weight:600;">
                Home
            </a>
        </div>
    """, unsafe_allow_html=True)

    # --------------------------
    # FLOATING AGrinEXT BUTTON (BOTTOM RIGHT)
    # --------------------------
    st.markdown("""
        <style>
        .agrinext-btn {
            position: fixed;
            bottom: 15px;
            right: 15px;
            background: #2c7be5;
            color: white;
            padding: 12px 22px;
            border-radius: 30px;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            z-index: 9999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .agrinext-btn:hover {
            background: #1a68d1;
        }
        </style>

        <a href="https://rahul-9307.github.io/AgriNext/" target="_blank" class="agrinext-btn">
            AgriNext
        </a>
    """, unsafe_allow_html=True)

    # --------------------------
    # MAIN TITLE
    # --------------------------
    st.title("🌱 SMART CROP RECOMMENDATION SYSTEM")

    # --------------------------
    # SIDEBAR INPUTS
    # --------------------------
    st.sidebar.header("Enter Soil & Weather Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 150.0, 0.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 150.0, 0.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 200.0, 0.0)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 0.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0)
    ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

    # --------------------------
    # PREDICT BUTTON
    # --------------------------
    if st.sidebar.button("Predict Crop"):
        result = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌾 Recommended Crop: **{result}**")


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    main()
