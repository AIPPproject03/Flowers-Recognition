import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Flower Recognition", layout="wide")

# CSS to improve the interface
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #4CAF50;
    text-align: center;
}
.sub-header {
    font-size: 1.5rem;
    color: #2E7D32;
}
.prediction-text {
    font-size: 1.8rem;
    font-weight: bold;
    color: #1565C0;
}
.confidence-text {
    font-size: 1.2rem;
    color: #0D47A1;
}
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = "models/flower_classifier_densenet.h5"
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
FLOWER_EMOJIS = {"daisy": "🌼", "dandelion": "🌞", "rose": "🌹", "sunflower": "🌻", "tulip": "🌷"}

@st.cache_resource
def load_model_from_file():
    """Load the pre-trained model"""
    return load_model(MODEL_PATH)

def predict_flower(img, model):
    """Make prediction on the uploaded image"""
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    
    return class_idx, confidence, preds

def main():
    st.markdown("<h1 class='main-header'>Flower Recognition App</h1>", unsafe_allow_html=True)
    st.write("Upload a flower image to classify it as daisy, dandelion, rose, sunflower, or tulip")
    
    # Load the model
    try:
        model = load_model_from_file()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model file exists in the 'models' directory")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])
    
    # Sample images
    st.markdown("<h3 class='sub-header'>Or try a sample image:</h3>", unsafe_allow_html=True)
    
    # Get sample images from each class
    samples = {}
    cols = st.columns(5)
    
    for i, class_name in enumerate(CLASS_NAMES):
        sample_path = os.path.join("flowers", class_name)
        try:
            # Get the first image from each class folder
            img_path = os.path.join(sample_path, os.listdir(sample_path)[0])
            samples[class_name] = img_path
            
            # Display sample images with buttons - fixed deprecated parameter
            with cols[i]:
                st.image(img_path, caption=f"{FLOWER_EMOJIS[class_name]} {class_name.title()}", width=120)
                if st.button(f"Try {class_name.title()}"):
                    uploaded_file = img_path
        except Exception as e:
            st.error(f"Could not load sample for {class_name}: {e}")
    
    if uploaded_file is not None:
        # Handle both uploaded file and sample selection
        try:
            if isinstance(uploaded_file, str):  # Sample image path
                img = Image.open(uploaded_file)
                st.image(uploaded_file, caption="Selected Sample Image", width=300)
            else:  # Uploaded file
                img = Image.open(uploaded_file)
                st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            # Preprocess the image
            img_resized = img.resize(IMG_SIZE)
            
            # Make prediction
            with st.spinner('Analyzing the image...'):
                class_idx, confidence, all_preds = predict_flower(img_resized, model)
                
                # Display results with emoji
                flower_emoji = FLOWER_EMOJIS.get(CLASS_NAMES[class_idx], "")
                st.markdown(f"<p class='prediction-text'>Prediction: {flower_emoji} {CLASS_NAMES[class_idx].title()}</p>", 
                            unsafe_allow_html=True)
                st.markdown(f"<p class='confidence-text'>Confidence: {confidence:.2%}</p>", 
                            unsafe_allow_html=True)
                
                # Display bar chart of predictions
                st.subheader("Prediction Probabilities")
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(CLASS_NAMES, all_preds[0] * 100, color='skyblue')
                
                # Highlight the predicted class
                bars[class_idx].set_color('green')
                
                ax.set_ylabel('Confidence (%)')
                ax.set_xlabel('Flower Type')
                ax.set_ylim([0, 100])
                plt.xticks(rotation=45)
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This app uses a fine-tuned DenseNet121 model that was pre-trained on ImageNet and then 
        fine-tuned on a dataset of flower images. The model can recognize 5 types of flowers:
        
        - 🌼 Daisy
        - 🌞 Dandelion
        - 🌹 Rose
        - 🌻 Sunflower
        - 🌷 Tulip
        
        The model was trained with data augmentation techniques to improve its generalization capability.
        """)

if __name__ == "__main__":
    main()