import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
# Load YOLOv8 segmentation model

model = YOLO('best_seg_LungNodule.pt')

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

image = Image.open('i3l_logo.png')

col1, col2 = st.columns([1,3])
with col1:
    st.image(image)
with col2:
    st.title("i3L AI-based Gram Staining Detection")

# Streamlit UI
st.title("Lung Nodule Segmentation")

uploaded_file = st.file_uploader("Upload an Image", 
    type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, 
        caption="Uploaded Image", 
        use_container_width=True)

    if st.button("Run Segmentation"):
        # Convert to numpy array
        img_np = np.array(image)

        # Run prediction
        results = model.predict(img_np)[0]

        # Visualize the mask overlay
        seg_img = results.plot()  # returns a numpy array with the segmentation mask overlaid

        # Show result
        st.image(seg_img, 
            caption="Segmentation Result", 
            use_container_width=True)

