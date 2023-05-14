import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import time
import os

# Set class names used in the model
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


@st.cache_resource
def load_model(path):
    """
    Parameters:
    path - takes path of the model to be loaded

    Returns-
    A loaded model.
    """
    model = tf.keras.models.load_model(path)
    return model


@st.cache_data
def predict(path):
    """
    Parameters:
    path - takes path of an image which is input to the model

    Returns-
    A score for a class predicted by the model.
    """
    model = load_model("model/brainT_detect.h5")
    img = tf.keras.utils.load_img(path, target_size=(180, 180))
    input_arr = tf.keras.utils.img_to_array(img)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    scores = tf.nn.softmax(predictions[:])

    return scores


@st.cache_data
def make_path(option):
    return os.path.join("img/test_imgs/", option)


def prediction_runner(img_path):
    img = Image.open(img_path)
    resized_img = img.resize((256, 256))
    st.title("Here is the image you've selected:")
    st.divider()
    st.image(resized_img)
    st.divider()

    scores = predict(img_path)
    st.title("Result of MRI scan:")

    with st.spinner(text="In progress..."):
        time.sleep(3)
    if (class_names[np.argmax(scores)]) == "No Tumor":
        st.success(
            f"This image most likely belongs to '{class_names[np.argmax(scores)]}' with a {100 * np.max(scores):.2f} percent confidence.",
            icon="✅",
        )
    else:
        st.error(
            f"This image most likely belongs to '{class_names[np.argmax(scores)]}' with a {100 * np.max(scores):.2f} percent confidence.",
            icon="❗",
        )
