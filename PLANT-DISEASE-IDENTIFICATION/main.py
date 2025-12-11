import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="AgriSens – Smart Disease Detection", layout="centered")


# -------------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"

    if os.path.exists(model_path):
        st.success("✅ Model Loaded Successfully!")
        return tf.keras.models.load_model(model_path)

    st.error("❌ Model file NOT FOUND! Please place trained_plant_disease_model.keras in same folder.")
    return None


model = load_model()


# -------------------------------------------------------
# PREDICTION FUNCTION (ACCURATE)
# -------------------------------------------------------
def model_prediction(uploaded_image):
    """Run image preprocessing and prediction"""

    # Save uploaded file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Load and preprocess
    image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # Normalize
    input_arr = input_arr / 255.0

    # Expand dims for batch shape
    input_arr = np.expand_dims(input_arr, axis=0)

    # Predict
    predictions = model.predict(input_arr)
    return int(np.argmax(predictions))


# -------------------------------------------------------
# CLASS LABELS
# -------------------------------------------------------
class_names = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]


# -------------------------------------------------------
# DISEASE INFO FOR CARD
# -------------------------------------------------------
disease_info = {
    "Apple___Apple_scab": {
        "sym": "Dark, velvety, irregular spots on leaves and fruits.",
        "treat": "Spray Mancozeb or Captan every 10–14 days."
    },
    "Potato___Late_blight": {
        "sym": "Water-soaked brown spots on leaves spreading rapidly.",
        "treat": "Spray Metalaxyl + Mancozeb immediately."
    },
    "Tomato___Early_blight": {
        "sym": "Concentric brown rings on lower leaves.",
        "treat": "Apply Chlorothalonil or Copper Oxychloride."
    }
}


# -------------------------------------------------------
# SIDEBAR PAGE NAVIGATION
# -------------------------------------------------------
st.sidebar.title("🌿 AgriSens")
page = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])


# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if page == "HOME":
    st.markdown("<h1 style='text-align:center;color:#2ecc71;'>SMART DISEASE DETECTION</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; font-size:18px;'>
    AI-based system to detect plant diseases using leaf images.<br>
    Fast, accurate and farmer-friendly.
    </p>
    """, unsafe_allow_html=True)


# -------------------------------------------------------
# DISEASE RECOGNITION PAGE
# -------------------------------------------------------
elif page == "DISEASE RECOGNITION":

    st.header("🌱 Disease Recognition")
    uploaded_image = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):

        if uploaded_image is None:
            st.error("Please upload an image first!")
        else:
            st.info("⏳ Processing your image...")

            index = model_prediction(uploaded_image)
            predicted_label = class_names[index]

            st.success(f"🌿 **Predicted Disease:** {predicted_label} ")

            # Show disease info card
            if predicted_label in disease_info:
                d = disease_info[predicted_label]

                st.markdown(f"""
                <div style="
                    background:#1e1e1e;
                    padding:20px;
                    border-radius:12px;
                    color:white;
                    margin-top:15px;
                    box-shadow:0 0 10px rgba(0,255,150,0.3);
                ">
                    <h3>📝 Symptoms</h3>
                    <p>{d['sym']}</p>

                    <h3>💊 Treatment</h3>
                    <p>{d['treat']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ No additional disease information available for this class.")


# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
Developed by <b>AgriSens Team</b> | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
