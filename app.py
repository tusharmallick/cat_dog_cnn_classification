import os
import warnings

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras")
DEFAULT_IMG_SIZE = 128          # used if the model's input size can't be read
GITHUB_URL = "https://github.com/tusharmallick/cat_dog_cnn_classification"

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        color: #222;               /* readable in both light and dark themes */
    }
    .result-box h3 { margin: 0 0 8px 0; color: #222; }
    .cat-result { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .dog-result { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
    .confidence { font-size: 1.3em; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained Keras model once and reuse it across reruns."""
    import tensorflow as tf  # imported lazily so import errors show in the UI

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    # compile=False is fine for inference-only use
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Read the expected input size from the model (falls back to the default)
    img_size = DEFAULT_IMG_SIZE
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if shape[1] and shape[2]:
            img_size = int(shape[1])
    except Exception:
        pass

    return model, img_size


try:
    model, IMG_SIZE = load_model()
except Exception as e:
    st.error(
        f"""
        ❌ **Error loading model:** {e}

        Please make sure:
        1. `best_model.keras` is in the same folder as `app.py`
        2. All dependencies from `requirements.txt` are installed
        3. The TensorFlow version matches the one used to train/save the model
        """
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Convert a PIL image into a (1, size, size, 3) float32 batch in [0, 1]."""
    img = ImageOps.exif_transpose(image)          # fix phone-photo rotation
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_dog_probability(image: Image.Image) -> float:
    """Return P(dog) as a float in [0, 1]."""
    batch = preprocess(image, IMG_SIZE)
    pred = model.predict(batch, verbose=0)
    return float(np.clip(np.ravel(pred)[0], 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Header + sidebar
# --------------------------------------------------------------------------- #
st.markdown('<div class="main-header">🐾 Cats vs Dogs Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transfer Learning CNN using TensorFlow</div>', unsafe_allow_html=True)

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(
    f"""
- **Architecture**: Transfer Learning CNN
- **Input Size**: {IMG_SIZE}×{IMG_SIZE} pixels
- **Classes**: Cat | Dog
- **Framework**: TensorFlow
- **Model File**: best_model.keras
"""
)

# --------------------------------------------------------------------------- #
# Image input
# --------------------------------------------------------------------------- #
st.markdown("### 📸 Upload an Image")

upload_method = st.radio("Choose input method:", ["Upload Image", "Use Camera"], horizontal=True)

if upload_method == "Upload Image":
    source_file = st.file_uploader(
        "Choose an image file (JPG, PNG, etc.)",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        help="Upload an image of a cat or dog for classification",
    )
else:
    source_file = st.camera_input("Take a photo")

image = None
if source_file is not None:
    try:
        image = Image.open(source_file)
        image.load()  # force decoding now so bad files fail here
    except Exception as e:
        st.error(f"❌ Could not open this image: {e}")
        image = None

# --------------------------------------------------------------------------- #
# Prediction UI
# --------------------------------------------------------------------------- #
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        st.markdown("#### Image Details")
        st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
        st.write(f"**Format:** {image.format or 'N/A'}")
        st.write(f"**Mode:** {image.mode}")

    if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                dog_prob = predict_dog_probability(image)
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                st.stop()

        cat_prob = 1.0 - dog_prob

        if dog_prob >= 0.5:
            label, emoji, css_class, conf = "Dog", "🐕", "dog-result", dog_prob
        else:
            label, emoji, css_class, conf = "Cat", "🐱", "cat-result", cat_prob

        st.markdown("---")

        # Whole box rendered in ONE markdown call so the styling actually applies
        st.markdown(
            f"""
            <div class="result-box {css_class}">
                <h3>{emoji} {label}</h3>
                <div class="confidence">Confidence: {conf * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Dog Probability", f"{dog_prob * 100:.1f}%")
        m2.metric("Cat Probability", f"{cat_prob * 100:.1f}%")

        st.markdown("### Prediction Breakdown")
        b1, b2 = st.columns(2)
        with b1:
            st.progress(dog_prob, text=f"Dog: {dog_prob:.1%}")
        with b2:
            st.progress(cat_prob, text=f"Cat: {cat_prob:.1%}")

        if conf >= 0.9:
            st.success(f"✅ High confidence: This is very likely a **{label}**.")
        elif conf >= 0.7:
            st.info(f"✓ Good confidence: This appears to be a **{label}**.")
        elif conf >= 0.6:
            st.warning(f"⚠️ Moderate confidence: Likely a **{label}**, but uncertain.")
        else:
            st.warning("⚠️ Low confidence: The image may be ambiguous or low quality.")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        <p>🔬 Built with Transfer Learning CNN | TensorFlow & Streamlit</p>
        <p><a href='{GITHUB_URL}' target='_blank'>View Project on GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
