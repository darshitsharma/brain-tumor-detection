import streamlit as st
import tensorflow as tf
import os
from helper import predict, make_path, prediction_runner


if __name__ == "__main__":
    # Page configurations
    st.set_page_config(
        page_title="NeuralCare",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    # Set title of webpage
    st.title("Brain Tumour Detection")

    instructions = """
	Welcome! This app lets you detect brain tumour in a patient's MRI image. All you have to do is to provide an MRI image. You can also select images from the sidebar to get a pre-configured image. The image you select or upload is directly feeded to a deep neural network in real-time and the output is displayed on your screen.
	"""
    st.write(instructions)

    # # Make option for user to select per-configure image or upload their own
    # option = st.radio(
    #     "How would you like to proceed?",
    #     ["Select a pre-configured image", "Upload your own image"],
    # )

    # Provide file uploader widget
    file = st.file_uploader("Upload an Image")

    # If user uploads data
    if file:
        prediction_runner(img_path=file)

    else:
        # Construct a sidebar
        st.sidebar.title("Select a pre-configured image 👇🏻")

        # Access all classes from the directory
        test_imgs = os.listdir("img/test_imgs/")
        test_imgs = list(map(lambda x: x.title(), test_imgs))
        selected_option = st.sidebar.selectbox("", test_imgs)
        selected_option = selected_option.lower()
        # Load images from the selected class
        available_images = os.listdir(make_path(selected_option))
        image_name = st.sidebar.selectbox("Image Name", available_images)
        img_path = os.path.join(make_path(selected_option) + "/" + image_name)

        # Make predictions
        prediction_runner(img_path)

    st.divider()
    st.info(
        "The neural network is trained only on brain's MRI images. Uploading any other image will give false and absurd results. 🚫"
    )
