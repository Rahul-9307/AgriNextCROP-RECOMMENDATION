import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

# ---- Load image safely ----
def load_image(image_name):
    image_path = os.path.join(os.path.dirname(__file__), image_name)
    return Image.open(image_path)

img = load_image("crop.png")
st.image(img, use_column_width=True)

# ---- Load Dataset ----
df = pd.read_csv("Crop_recommendation.csv")

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ---- Train Model (simple & small) ----
model = RandomForestClassifier(n_estimators=20, random_state=5)
model.fit(X, y)

# ---- Prediction Function ----
def predict_crop(n, p, k, temp, hum, ph, rain):
    data = np.array([[n, p, k, temp, hum, ph, rain]])
    return model.predict(data)[0]

# ---- Streamlit UI ----
def main():
    st.title("🌱 SMART CROP RECOMMENDATION SYSTEM")

    st.sidebar.header("Enter Soil & Environment Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 150.0, 0.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 150.0, 0.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 200.0, 0.0)
    temperature = st.sidebar.number_input("Temperature °C", 0.0, 50.0, 0.0)
    humidity = st.sidebar.number_input("Humidity %", 0.0, 100.0, 0.0)
    ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

    if st.sidebar.button("Predict Crop"):
        result = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌾 Recommended Crop: **{result}**")

if __name__ == "__main__":
    main()
