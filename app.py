import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os

model_file = "weather_cnn_best.h5"
gdrive_file_id = "1nv1NVvsmdgmn5iPpf-3zAGNje8XdAwy3"

# Only download if it doesn't exist yet
if not os.path.exists(model_file):
    gdown.download(id=gdrive_file_id, output=model_file, quiet=False)

# Load the model
model = load_model(model_file)

# Class labels
class_labels = ['shine', 'sunrise', 'cloudy', 'rain']

# Streamlit UI
st.title("🌤️ Weather Image Classifier")
st.write("Upload an image of the weather, and the model will predict the weather type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((100, 100))  # Adjust based on your model's expected input
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize if your model was trained this way
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"Predicted Weather: **{predicted_class.capitalize()}**")
