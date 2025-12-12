import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG (centered)
# -----------------------------------------------------------
st.set_page_config(page_title="Agri🌾Next", layout="centered")

# -----------------------------------------------------------
# RAW IMAGE LINKS
# -----------------------------------------------------------
HERO_IMAGE = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Diseases.png"

IMG_REALTIME = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Real-Time%20Results.png"
IMG_INSIGHTS = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Actionable%20Insights.png"
IMG_DETECTION = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Disease%20Detection.png"

# -----------------------------------------------------------
# CUSTOM CSS (Center + Compact)
# -----------------------------------------------------------
st.markdown("""
<style>

.block-container {
    max-width: 900px;
    padding-top: 10px;
}

/* HERO IMAGE CONTAINER */
.hero-wrapper {
    display: flex;
    justify-content: center;
}
.hero-img {
    width: 95%;
    border-radius: 16px;
    border: 2px solid #2ecc71;
    box-shadow: 0px 0px 10px rgba(0,255,140,0.20);
}

.center-text { text-align: center; }

.app-footer {
    background:#111;
    padding:10px;
    border-radius:8px;
    margin-top:20px;
    text-align:center;
    font-size:13px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# HERO SECTION (Big + Centered)
# -----------------------------------------------------------
st.markdown(f"""
<div class='hero-wrapper'>
    <img src='{HERO_IMAGE}' class='hero-img'>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# PAGE SELECTOR (Centered)
# -----------------------------------------------------------
cols = st.columns([1, 2, 1])
with cols[1]:
    page = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])

# -----------------------------------------------------------
# CLASS LABELS
# -----------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# -----------------------------------------------------------
# MODEL LOADING (Auto Detect)
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    target_name = "trained_plant_disease_model.keras"
    found_path = None

    for root, dirs, files in os.walk(".", topdown=True):
        if target_name in files:
            found_path = os.path.join(root, target_name)
            break

    st.write("🔍 Searching model...")

    if found_path:
        st.success(f"✅ Model Loaded: {found_path}")
        return tf.keras.models.load_model(found_path)

    st.error("❌ Model NOT FOUND! Please upload trained_plant_disease_model.keras.")
    return None


model = load_model()

# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, 0)
    pred = model.predict(arr)
    idx = np.argmax(pred)
    return idx, CLASS_NAMES[idx], float(np.max(pred))


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "HOME":
    st.markdown("<h1 class='center-text' style='color:#2ecc71;'>Agri🌾Next: Smart Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center-text' style='color:#ccc;'>AI-powered platform for accurate plant disease recognition.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(IMG_REALTIME, width=220)
        st.markdown("<p class='center-text'><b>Real-Time Results</b></p>", unsafe_allow_html=True)
    with c2:
        st.image(IMG_INSIGHTS, width=220)
        st.markdown("<p class='center-text'><b>Actionable Insights</b></p>", unsafe_allow_html=True)
    with c3:
        st.image(IMG_DETECTION, width=220)
        st.markdown("<p class='center-text'><b>Disease Detection</b></p>", unsafe_allow_html=True)


# -----------------------------------------------------------
# DISEASE RECOGNITION PAGE
# -----------------------------------------------------------
elif page == "DISEASE RECOGNITION":
    st.markdown("<h2 class='center-text' style='color:#2ecc71;'>🌿 Disease Recognition</h2>", unsafe_allow_html=True)

    # Center uploader
    a, b, c = st.columns([1, 2, 1])
    with b:
        uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image(uploaded, width=350)
        st.markdown("</div>", unsafe_allow_html=True)

        temp = "uploaded_temp.jpg"
        with open(temp, "wb") as f:
            f.write(uploaded.getbuffer())

        btn_row = st.columns([1, 1, 1])
        with btn_row[1]:
            detect = st.button("🔍 Detect Disease")

        if detect:
            if model is None:
                st.error("❌ Model not loaded!")
            else:
                idx, disease, conf = predict_image(temp)

                st.markdown(f"""
                <div style='border:1px solid #2ecc71; padding:12px; border-radius:10px; max-width:450px; margin:auto; text-align:center;'>
                    <h3 style='color:#2ecc71;'>🌱 Predicted: {disease}</h3>
                    <p style='color:#ccc;'>Confidence: {conf*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("<div class='app-footer'>Developed by <b>Team Agri🌾Next</b> | Powered by Streamlit</div>", unsafe_allow_html=True)
