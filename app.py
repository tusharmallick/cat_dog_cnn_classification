"""
Streamlit deployment app for the Cats vs Dogs transfer-learning model.

Run with:
    pip install streamlit tensorflow pillow numpy
    streamlit run app.py

Expects a saved model file named "best_model.keras" in the same folder
(produced by the last cell of the training notebook).
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "best_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Cat", "Dog"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # model has its own Rescaling/preprocess layer inside


def main():
    st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
    st.title("🐾 Cat vs Dog Classifier")
    st.write("Upload an image and the model will predict whether it's a cat or a dog.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Predicting..."):
            x = preprocess(image)
            prob = float(model.predict(x, verbose=0).ravel()[0])
            pred_idx = int(prob >= 0.5)
            pred_label = CLASS_NAMES[pred_idx]
            confidence = prob if pred_idx == 1 else 1 - prob

        st.subheader(f"Prediction: **{pred_label}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.progress(confidence)


if __name__ == "__main__":
    main()
