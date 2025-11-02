# app.py
"""
Enhanced Streamlit app for Emotion Detection (HOG + SVM)
Run with: streamlit run app.py
"""
import streamlit as st
from skimage import color, transform
from skimage.feature import hog
import numpy as np
import joblib
from PIL import Image
import io as sysio
import cv2

MODEL_PATH = "models/emotion_svm.pkl"
IMAGE_SIZE = (48, 48)

st.set_page_config(page_title="Emotion Detector", page_icon="🧠", layout="centered")

# --- FUNCTIONS ---
@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"]

def preprocess_image(image_bytes):
    img = Image.open(sysio.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    fd = hog(arr, orientations=9, pixels_per_cell=(8,8),
             cells_per_block=(2,2), block_norm='L2-Hys')
    return fd.reshape(1, -1)

def preprocess_pil_image(pil_img):
    img = pil_img.convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    fd = hog(arr, orientations=9, pixels_per_cell=(8,8),
             cells_per_block=(2,2), block_norm='L2-Hys')
    return fd.reshape(1, -1)

# --- HEADER ---
st.markdown("<h1 style='text-align:center;'>Emotion Detector (HOG + SVM) 💫</h1>", unsafe_allow_html=True)
st.write("Upload a face image or take a live photo — the model will predict the detected emotion.")

# --- MODEL LOAD ---
try:
    model = load_model()
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error("⚠️ Model not found. Please run `python model.py` to train and create models/emotion_svm.pkl.")
    st.stop()

# --- IMAGE INPUT OPTIONS ---
st.divider()
choice = st.radio("Select input method:", ["📁 Upload Image", "📸 Take Photo"], horizontal=True)
image_data = None

if choice == "📁 Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image_data = uploaded.read()
        st.image(image_data, caption="Uploaded Image", width=300)

elif choice == "📸 Take Photo":
    cam_img = st.camera_input("Take a live photo")
    if cam_img is not None:
        image_data = cam_img.getvalue()
        st.image(image_data, caption="Captured Image", width=300)

# --- PREDICTION ---
if image_data is not None:
    with st.spinner("Analyzing emotion... 🧠"):
        feats = preprocess_image(image_data)
        pred = model.predict(feats)[0]

    st.success(f"### 🧍 Detected Emotion: **{pred}**")
    st.balloons()

# --- FOOTER ---
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed with ❤️ by [Your Name] | Powered by Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
