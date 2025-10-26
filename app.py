import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import os
import zipfile
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(
    page_title="Skin Cancer AI Detector",
    page_icon="\U0001F9EC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesions'
}

# Risk levels with colors
RISK_LEVELS = {
    'mel': ('HIGH RISK', 'error'),
    'bcc': ('HIGH RISK', 'error'),
    'akiec': ('MODERATE RISK', 'warning'),
    'bkl': ('MODERATE RISK', 'warning'),
    'df': ('LOW RISK', 'success'),
    'nv': ('LOW RISK', 'success'),
    'vasc': ('LOW RISK', 'success')
}

# Sorted by risk: LOW → MODERATE → HIGH
RISK_ORDER = {
    'LOW RISK': ['df', 'nv', 'vasc'],
    'MODERATE RISK': ['akiec', 'bkl'],
    'HIGH RISK': ['bcc', 'mel']
}

# Model config
MODEL_PATH = 'efficientnetb3_ham10000_v1.keras'
MODEL_ZIP = 'efficientnetb3_ham10000_model.zip'
HUGGINGFACE_REPO = 'foroughm423/skin-cancer-efficientnetb3'
HUGGINGFACE_FILE = 'efficientnetb3_ham10000_model.zip'

# SECURE TOKEN HANDLING
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))

if not HUGGINGFACE_TOKEN:
    st.error("HUGGINGFACE_TOKEN not found. Set it in Streamlit Secrets or environment.")
    st.stop()

# Focal Loss
def focal_loss(gamma=2.0, alpha=None):
    if alpha is None:
        alpha = [0.5, 0.3, 0.3, 1.2, 0.55, 0.25, 1.5]
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_true_one_hot = tf.one_hot(y_true, depth=7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        alpha_t = tf.reduce_sum(y_true_one_hot * alpha_tensor, axis=-1, keepdims=True)
        ce = -y_true_one_hot * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        fl = alpha_t * weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return focal_loss_fixed

# Load model (clean UI)
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})

    placeholder = st.empty()
    with placeholder.container():
        with st.spinner('Downloading model from Hugging Face (~45 MB)...'):
            try:
                downloaded_path = hf_hub_download(
                    repo_id=HUGGINGFACE_REPO,
                    filename=HUGGINGFACE_FILE,
                    local_dir='.',
                    token=HUGGINGFACE_TOKEN,
                    local_dir_use_symlinks=False
                )
                if downloaded_path != MODEL_ZIP:
                    if os.path.exists(MODEL_ZIP):
                        os.remove(MODEL_ZIP)
                    os.rename(downloaded_path, MODEL_ZIP)
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                return None

        with st.spinner('Extracting model...'):
            try:
                with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                    zip_ref.extractall('.')
                os.remove(MODEL_ZIP)
            except Exception as e:
                st.error(f"Extraction failed: {str(e)}")
                return None

        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            return None

        with st.spinner('Loading model into memory...'):
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})

    placeholder.empty()
    return model

# UI
st.title("Skin Cancer AI Detector")
st.markdown("*AI-powered classification of 7 skin lesion types using EfficientNetB3*")

# Sidebar
with st.sidebar:
    st.header("Model Info")
    st.metric("Test Accuracy", "77.13%")
    st.metric("Melanoma Recall", "63.68%")
    st.caption("**Dataset:** HAM10000 (10,015 images)")
    st.caption("**Training:** 45 epochs, Focal Loss")

    st.markdown("---")
    st.subheader("Detectable Lesions")

    # LOW RISK (Green)
    st.success("**LOW RISK**")
    for cls in RISK_ORDER['LOW RISK']:
        st.markdown(f"• **{CLASS_NAMES[cls]}**")

    # MODERATE RISK (Yellow)
    st.warning("**MODERATE RISK**")
    for cls in RISK_ORDER['MODERATE RISK']:
        st.markdown(f"• **{CLASS_NAMES[cls]}**")

    # HIGH RISK (Red)
    st.error("**HIGH RISK**")
    for cls in RISK_ORDER['HIGH RISK']:
        st.markdown(f"• **{CLASS_NAMES[cls]}**")

    st.markdown("---")
    st.caption("**Disclaimer:** Educational tool only. Consult a dermatologist.")

# Load model
model = load_model()
if model is None:
    st.error("Unable to load model. Please refresh.")
    st.stop()

# Upload
uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if uploaded_file:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

    # Preprocess
    img_resized = image.resize((300, 300))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array, verbose=0)[0]

    top_idx = np.argmax(predictions)
    top_class = CLASSES[top_idx]
    confidence = predictions[top_idx] * 100

    with col2:
        st.subheader("Prediction")
        st.metric("**Diagnosis**", CLASS_NAMES[top_class])
        st.metric("**Confidence**", f"{confidence:.2f}%")
        st.progress(float(confidence / 100))

        risk_text, risk_style = RISK_LEVELS[top_class]
        if risk_style == 'error':
            st.error(f"**{risk_text}**")
            st.write("Immediate medical attention recommended.")
        elif risk_style == 'warning':
            st.warning(f"**{risk_text}**")
            st.write("Consult a dermatologist.")
        else:
            st.success(f"**{risk_text}**")
            st.write("Monitor for changes.")

    # Probabilities
    st.markdown("---")
    st.subheader("Classification Probabilities")

    sorted_idx = np.argsort(predictions)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        cls = CLASSES[idx]
        prob = predictions[idx] * 100
        col_r, col_n, col_b = st.columns([0.3, 1.5, 2])
        with col_r: st.write(f"**#{rank}**")
        with col_n:
            name = f"**{CLASS_NAMES[cls]}** Target" if idx == top_idx else CLASS_NAMES[cls]
            st.write(name)
        with col_b:
            st.progress(float(prob / 100), text=f"{prob:.2f}%")

    st.markdown("---")
    st.warning("**IMPORTANT:** This is a research tool. Always consult a qualified dermatologist.")

else:
    st.info("Upload an image to begin analysis.")
    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "77.13%")
    c2.metric("Melanoma Recall", "63.68%")
    c3.metric("Macro AUC", "0.94")
    c4.metric("Dataset", "10,015")

    # Performance visualizations
    st.markdown("---")
    st.subheader("Model Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curves", "Training History"])

    with tab1:
        if os.path.exists("./images/confusion_matrix.png"):
            st.image("./images/confusion_matrix.png", caption="Classification performance per lesion type")
        else:
            st.info("Visualization not available")

    with tab2:
        if os.path.exists("./images/roc_curves.png"):
            st.image("./images/roc_curves.png", caption="ROC curves for all classes")
        else:
            st.info("Visualization not available")

    with tab3:
        if os.path.exists("./images/training_history.png"):
            st.image("./images/training_history.png", caption="Training progress over 45 epochs")
        else:
            st.info("Visualization not available")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Built with <b>TensorFlow 2.19</b> & <b>Streamlit</b> | 
        <a href='https://github.com/foroughm423'>GitHub</a> | 
        <a href='https://kaggle.com/foroughgh95'>Kaggle</a><br>
        © 2025 | Research Project
    </div>
    """,
    unsafe_allow_html=True
)