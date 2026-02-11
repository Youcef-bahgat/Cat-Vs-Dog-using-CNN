import streamlit as st
import tensorflow as tf
import numpy as np
import keras 
from PIL import Image

# تحميل الموديل
model = keras.models.load_model("model.h5", compile=False)

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image and the model will predict Cat or Dog")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success(f"🐶 Dog — Confidence: {prediction:.2f}")
    else:
        st.success(f"🐱 Cat — Confidence: {1-prediction:.2f}")

