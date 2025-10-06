import streamlit as st
from PIL import Image
import numpy as np
import io
from ultralytics import YOLO
import cv2

# Load YOLOv8 segmentation model
model = YOLO('best_seg_LungNodule.pt')

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

# Logo + Title
image = Image.open('i3l_logo.png')
col1, col2 = st.columns([1, 3])
with col1:
    st.image(image, use_container_width=True)
with col2:
    st.title("i3L AI-based Lung Nodule Segmentation")

st.header("Lung Nodule Segmentation")

# Upload UI
uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")

    # Two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image2, caption="Original Image", use_container_width=True)

    with col2:
        seg_img_placeholder = st.empty()

    # Transparency slider
    alpha = st.slider("🩶 Adjust Segmentation Transparency", 0.0, 1.0, 0.5, 0.05)

    # Centered "Run Segmentation" button
    st.markdown("<hr>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_clicked = st.button("Run Segmentation", use_container_width=True)

    if run_clicked:
        with st.spinner("Running segmentation..."):
            img_np = np.array(image2)
            results = model.predict(img_np, verbose=False)[0]
            seg_img = results.plot()  # mask overlay on image

            # Blend original image + segmentation with alpha transparency
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
            blended = cv2.addWeighted(np.array(image2), 1 - alpha, seg_img, alpha, 0)

        # Show blended result
        seg_img_placeholder.image(blended, 
                                  caption=f"Segmentation Result (Transparency: {alpha:.2f})", 
                                  use_container_width=True)

        # Convert to bytes for download
        seg_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        seg_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download button
        st.download_button(
            label="Download Segmentation Result",
            data=byte_im,
            file_name="lung_nodule_segmentation.png",
            mime="image/png",
            use_container_width=True
        )
