import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os

model_file = "weather_cnn_best.h5"
gdrive_file_id = "1nv1NVvsmdgmn5iPpf-3zAGNje8XdAwy3"

# Download the model from Google Drive if not present
if not os.path.exists(model_file):
    gdown.download(id=gdrive_file_id, output=model_file, quiet=False)

# Load the trained model
model = load_model(model_file)

# Class labels in the same order as training
class_labels = ['Sunrise', 'Shine', 'Rain', 'Cloudy']

# Streamlit UI
st.title("🌤️ Weather Image Classifier")
st.write("Upload an image of the weather, and the model will predict the weather type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and convert to RGB (ensure consistency)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image to 100x100 (model input size)
    image = image.resize((100, 100))

    # Convert image to numpy array and normalize pixels to [0, 1]
    image_array = img_to_array(image).astype('float32') / 255.0

    # Expand dims to create batch size of 1
    image_array = np.expand_dims(image_array, axis=0)

    # Predict with the model
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Predicted Weather: **{predicted_class}** ({confidence*100:.2f}%)")
