import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .cat-result {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .dog-result {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .confidence {
        font-size: 1.3em;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    try:
        model = tf.keras.models.load_model("best_model.keras", compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except FileNotFoundError:
        st.error("""
        ❌ Model file not found! Please ensure `best_model.keras` is in the same directory as `app.py`.
        
        Download it from: https://github.com/tusharmallick/cat_dog_cnn_classification
        """)
        st.stop()

# Load model
model = load_model()

# App header
st.markdown('<div class="main-header">🐾 Cats vs Dogs Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transfer Learning CNN using TensorFlow</div>', unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
- **Architecture**: Transfer Learning CNN
- **Backbones Tested**: 5 pretrained models
- **Input Size**: 128×128 pixels
- **Classes**: Cat (0) | Dog (1)
- **Best Model**: Fine-tuned with EarlyStopping
""")

# Main content
st.markdown("### 📸 Upload an Image")

# Upload method selection
upload_method = st.radio("Choose upload method:", ["Upload Image", "Use Camera"], horizontal=True)

uploaded_file = None
camera_image = None

if upload_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, etc.)",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        help="Upload an image of a cat or dog for classification"
    )
else:
    camera_image = st.camera_input("Take a photo")

# Process uploaded or camera image
image_to_predict = None
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file)
elif camera_image is not None:
    image_to_predict = Image.open(camera_image)

if image_to_predict is not None:
    # Display image info
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_to_predict, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown("#### Image Details")
        st.write(f"**Size:** {image_to_predict.size[0]} × {image_to_predict.size[1]} px")
        st.write(f"**Format:** {image_to_predict.format}")
        st.write(f"**Mode:** {image_to_predict.mode}")
    
    # Preprocess image
    if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Resize to model input size
                img_array = image_to_predict.convert("RGB")
                img_array = img_array.resize((128, 128))
                img_array = np.array(img_array, dtype="float32") / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array, verbose=0)
                confidence = float(prediction[0][0])
                
                # Classify based on threshold
                if confidence >= 0.5:
                    prediction_class = "🐕 Dog"
                    prediction_label = "Dog"
                    prediction_confidence = confidence
                    css_class = "dog-result"
                else:
                    prediction_class = "🐱 Cat"
                    prediction_label = "Cat"
                    prediction_confidence = 1 - confidence
                    css_class = "cat-result"
                
                # Display result
                st.markdown("---")
                st.markdown(f'<div class="result-box {css_class}">', unsafe_allow_html=True)
                st.markdown(f'### {prediction_class}')
                st.markdown(f'<div class="confidence">Confidence: {prediction_confidence * 100:.2f}%</div>', unsafe_allow_html=True)
                
                # Detailed metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dog Probability", f"{confidence * 100:.1f}%")
                with col2:
                    st.metric("Cat Probability", f"{(1 - confidence) * 100:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown("### Prediction Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.progress(confidence, text=f"Dog: {confidence:.1%}")
                with col2:
                    st.progress(1 - confidence, text=f"Cat: {(1-confidence):.1%}")
                
                # Additional info
                if prediction_confidence >= 0.9:
                    st.success(f"✅ High confidence prediction: This is definitely a **{prediction_label}**!")
                elif prediction_confidence >= 0.7:
                    st.info(f"✓ Good confidence: This appears to be a **{prediction_label}**.")
                elif prediction_confidence >= 0.6:
                    st.warning(f"⚠️ Moderate confidence: This is likely a **{prediction_label}**, but uncertain.")
                else:
                    st.warning("⚠️ Low confidence: The model is uncertain. Image might be ambiguous or low quality.")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.error("Please ensure the image is valid and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>🔬 Built with Transfer Learning CNN | TensorFlow & Streamlit</p>
    <p><a href='https://github.com/tusharmallick/cat_dog_cnn_classification' target='_blank'>View Project on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
