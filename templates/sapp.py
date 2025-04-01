import streamlit as st
import numpy as np
import io
import os
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore
from PIL import Image

# Load the trained model
model_path = os.path.join(os.getcwd(), "artifacts", "training", "model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

model = load_model(model_path)

# Function to preprocess and predict image
def predict_image(img):
    # Convert image to RGB (to ensure 3 channels)
    img = img.convert("RGB")

    # Resize and convert to array
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch

    # Make prediction
    prediction = model.predict(img_array)
    
    # Determine binary or multi-class classification
    if model.output_shape[-1] == 1:  # Binary classification (sigmoid activation)
        result = (prediction > 0.5).astype(int)
    else:  # Multi-class classification (softmax activation)
        result = np.argmax(prediction, axis=1)

    return "Tumor" if result[0] == 1 else "No Tumor"

# Streamlit UI
def main():
    st.title("Tumor Classification App")
    st.write("Upload an image to classify if it contains a tumor.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Open and display image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)  # Fixed deprecation issue

        if st.button("Classify Image"):
            st.write("Classifying... Please wait.")
            prediction_result = predict_image(img)

            # Display result
            st.write("## Prediction Result:")
            st.write(f"🔍 **{prediction_result}**")

if __name__ == "__main__":
    main()
