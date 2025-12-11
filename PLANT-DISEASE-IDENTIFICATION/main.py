import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -----------------------------------------------------------
# LOAD MODEL (cached)
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    current_dir = os.path.dirname(__file__)

    # Try .keras file
    model_path = os.path.join(current_dir, "trained_plant_disease_model.keras")

    # Try .h5 if .keras not found
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, "trained_plant_disease_model.h5")

    # If still missing, return None
    if not os.path.exists(model_path):
        return None

    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    result = model.predict(arr)
    return np.argmax(result)


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AgriSens - Disease Detection", layout="centered")
st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🌾 SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)


# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.title("🌿 AgriSens")
page = st.sidebar.radio("Navigate", ["Home", "Disease Recognition"])


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "Home":
    st.markdown("""
        <div style='text-align:center; font-size:18px;'>
            Upload plant leaf images and detect disease using AI.<br>
            A simple, fast and accurate detection system for farmers.
        </div>
    """, unsafe_allow_html=True)

    image_path = os.path.join(os.path.dirname(__file__), "Diseases.png")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)


# -----------------------------------------------------------
# DISEASE RECOGNITION PAGE
# -----------------------------------------------------------
elif page == "Disease Recognition":

    st.markdown("<h2 style='text-align:center; color:#6A5ACD;'>📸 Upload Plant Leaf Image</h2>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, use_column_width=True, caption="Uploaded Image")

        # Save as temp file
        temp_path = "temp_leaf.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("🔍 Predict Disease"):
            if model is None:
                st.error("❌ Model file missing in GitHub folder!")
            else:
                st.info("⏳ Analyzing image... Please wait...")

                result_idx = predict_image(temp_path)

                # All classes
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot',
                    'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

                st.success(f"🌱 **Disease Identified: {class_name[result_idx]}**")


