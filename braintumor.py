import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image

# Function to load the brain tumor classification model
# @st.cacheresource  # Caches the model to improve performance
def loadbraintumormodel():
    # Load model architecture from JSON
    with open('resnet-50-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    model = tf.keras.models.model_from_json(json_savedModel)

    # Load weights into the model
    model.load_weights('weights.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

# Function to classify the uploaded image
def classify_brain_tumor(image_file):
    # Load the model
    model = loadbraintumormodel()

    # Preprocess the image
    img = Image.open(image_file)
    img = img.resize((224, 224))  # Resize to model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values

    # Make prediction
    predictions = model.predict(img_array)

    # Output raw prediction values
    st.write("Prediction Probabilities:", predictions[0])

def main():
    st.title("Brain Tumor Classification")

    # Upload image
    image_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        classify_brain_tumor(image_file)

