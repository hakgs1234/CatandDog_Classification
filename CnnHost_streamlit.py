import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
 
# Load the trained model
model = load_model("Image_classification_CNN.keras")

# Streamlit app title
st.title("Cat and Dog Image Classification")
st.subheader("Prediction Model by: Hamaad Ayub Khan")
# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Resize the image to match the input shape of the model
    resized_image = cv2.resize(opencv_image, (100, 100))

    # Display the image
    st.image(resized_image, channels="BGR")
    
    # Preprocess the image
    img_array = image.img_to_array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make predictions
    prediction = model.predict(img_array)
    prediction_class = (prediction > 0.5).astype(int)

    # Display the result
    if prediction_class == 0:
        st.write("The model predicts: Dog")
    else:
        st.write("The model predicts: Cat")

