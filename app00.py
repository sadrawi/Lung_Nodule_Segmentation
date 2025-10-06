import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('best_seg_LungNodule.pt')

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

# Logo + Title
image = Image.open('i3l_logo.png')
col1, col2 = st.columns([1,3])
with col1:
    st.image(image, use_container_width=True)
with col2:
    st.title("i3L AI-based Lung Nodule Segmentation")

# Upload UI
st.header("Lung Nodule Segmentation")
uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")

    # Create two equal-width columns for images
    col1, col2 = st.columns(2)

    with col1:
        st.image(image2, caption="Original Image", use_container_width=True)

    with col2:
        if st.button("Run Segmentation"):
            # Convert to numpy array
            img_np = np.array(image2)

            # Run YOLO segmentation
            results = model.predict(img_np, verbose=False)[0]

            # Overlay mask on image
            seg_img = results.plot()

            # Display side by side
            st.image(seg_img, caption="Segmentation Result", use_container_width=True)
