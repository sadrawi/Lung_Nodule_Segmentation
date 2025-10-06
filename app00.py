import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import io

# Load YOLOv8 segmentation model
model = YOLO('best_seg_LungNodule.pt')

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

# Logo and header
image = Image.open('i3l_logo.png')
col1, col2 = st.columns([1,3])
with col1:
    st.image(image)
with col2:
    st.title("i3L AI-based Lung Nodule Segmentation")

st.subheader("Lung Nodule Segmentation")

uploaded_file = st.file_uploader("Upload a Lung Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")

    # Convert to numpy for processing
    img_np = np.array(image2)

    if st.button("Run Segmentation"):
        # Run YOLO segmentation
        results = model.predict(img_np)[0]
        seg_img = results.plot()  # YOLO overlay

        # Convert BGR to RGB
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        # Transparency slider
        alpha = st.slider("Adjust Segmentation Transparency", 0.0, 1.0, 0.5)

        # Blend original and segmentation
        blended = cv2.addWeighted(img_np, 1 - alpha, seg_img, alpha, 0)

        # Show both
        st.image(img_np, caption="Original Image", use_container_width=True)
        st.image(blended, caption="Segmentation Overlay", use_container_width=True)

        # Prepare download
        blended_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        blended_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Result",
            data=byte_im,
            file_name="segmentation_result.png",
            mime="image/png"
        )
