import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image  # Rename to avoid conflict
import io

# Load the model
model_path = './model/cnn_model.h5'
model = load_model(model_path)

# Define class names
class_names = ['angular leaf spot', 'bean rust', 'healthy']
 
# Function to make predictionss
def predict(image):
    img = image.resize((224, 224))  # Resize the image
    img_array = keras_image.img_to_array(img)  # Use keras_image module
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class] 
    accuracy = np.max(prediction)
    return predicted_class_name, accuracy

# Streamlit app
def main():
    st.title("Bean Leaf Disease Classification")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        predicted_class, accuracy = predict(image)  # Pass the PIL Image object
        st.write("Prediction:", predicted_class)
        st.write("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
