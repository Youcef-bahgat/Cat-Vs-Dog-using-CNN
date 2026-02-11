# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -----------------------
# App title
st.title("Image Classification: Cat vs Dog")
st.write("Upload an image and the model will classify it!")

# -----------------------
# Load the model once
@st.cache_resource
def load_my_model():
    model = load_model("model.h5")  # change the name if needed
    return model

model = load_my_model()

# -----------------------
# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # -----------------------
    # Preprocess the image
    img = img.resize((128, 128))  # adjust according to your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = img_array / 255.0  # normalize if the model was trained on normalized images

    # -----------------------
    # Prediction
    prediction = model.predict(img_array)
    
    # -----------------------
    # Display the result
    # assuming binary classification: 0 = Cat, 1 = Dog
    if prediction[0][0] > 0.5:
        st.success("Result: 🐶 Dog")
    else:
        st.success("Result: 🐱 Cat")
