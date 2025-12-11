import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -----------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------
st.set_page_config(page_title="AgriSens – Smart Disease Detection", layout="wide")


# -----------------------------------------------------------
# GLOBAL CSS FOR FULL UI
# -----------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #0f1116 !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Poppins', sans-serif;
    color: #7CFF6B !important;
}

p, li {
    color: #d8d8d8;
    font-size: 18px;
}

.section-title {
    font-size: 45px;
    text-align: center;
    font-weight: 700;
    color: #7CFF6B;
    margin-top: 30px;
}

.feature-card {
    background: #171b22;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    transition: 0.3s ease;
    box-shadow: 0px 0px 10px rgba(0,255,150,0.08);
}

.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 0px 25px rgba(0,255,150,0.2);
}

.feature-title {
    font-size: 20px;
    color: #fff;
    margin-top: 10px;
}

.feature-desc {
    font-size: 16px;
    color: #bdbdbd;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    model_name = "trained_plant_disease_model.keras"

    for root, dirs, files in os.walk(".", topdown=True):
        if model_name in files:
            return tf.keras.models.load_model(os.path.join(root, model_name))

    st.error("❌ Model file NOT found! Please add trained_plant_disease_model.keras")
    return None


model = load_model()


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
    return np.argmax(model.predict(arr))


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
def home_page():

    # HERO IMAGE
    st.markdown("""
    <div style="width:100%; display:flex; justify-content:center;">
        <img src="https://i.ibb.co/3c2CZfL/plant-ai-banner.jpg"
             style="width:100%; max-width:1100px; border-radius:18px; margin-top:10px;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='section-title'>AgriSens: Smart Disease Detection</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; max-width:800px; margin:auto; font-size:20px;">
    Empowering Farmers with AI-Powered Plant Disease Recognition.<br>
    Upload plant images to detect diseases accurately and access actionable insights.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='section-title'>Features</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <img src="https://i.ibb.co/Jc2kzjF/leaf-scan.jpg" width="100%" style="border-radius:10px;">
            <div class="feature-title">Disease Detection</div>
            <div class="feature-desc">Identify plant diseases using AI.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <img src="https://i.ibb.co/ySmM1HG/farm-insight.jpg" width="100%" style="border-radius:10px;">
            <div class="feature-title">Actionable Insights</div>
            <div class="feature-desc">Get remedies and full disease details.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <img src="https://i.ibb.co/tYJ4QCQ/realtime-detection.jpg" width="100%" style="border-radius:10px;">
            <div class="feature-title">Real-Time Results</div>
            <div class="feature-desc">Instant prediction using neural networks.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 class='section-title'>How It Works</h2>", unsafe_allow_html=True)

    st.markdown("""
    <ol style="font-size:22px; max-width:800px; margin:auto; color:#fff;">
        <li>Select <b>Disease Recognition</b> from the sidebar.</li>
        <li>Upload an image of the plant leaf.</li>
        <li>Get instant AI-based disease results.</li>
    </ol>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------
# DISEASE PAGE
# -----------------------------------------------------------
def disease_page():

    st.title("🌱 Plant Disease Recognition")

    uploaded = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, use_column_width=True)

        path = "temp.jpg"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("🔍 Identify Disease"):

            idx = predict_image(path)

            class_list = [
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

            predicted = class_list[idx]

            st.success(f"🌿 Disease Detected: **{predicted}**")

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #6a11cb, #2575fc);
                padding: 25px;
                border-radius: 15px;
                color: white;
                margin-top: 20px;
            ">
                <h2 style="text-align:center;">🌿 Disease Report</h2>
                <p style="font-size:20px;">Detected Class: <b>{predicted}</b></p>
                <p>⚠ Full advisory content will display here.</p>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

if page == "Home":
    home_page()

else:
    disease_page()
