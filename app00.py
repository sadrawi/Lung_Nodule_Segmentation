import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import io

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

    # Placeholder for results
    col1, col2 = st.columns(2)

    with col1:
        st.image(image2, caption="Original Image", use_container_width=True)

    with col2:
        seg_img_placeholder = st.empty()  # reserve space for result later

    # Centered "Run Segmentation" button
    st.markdown("<hr>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_clicked = st.button("🔍 Run Segmentation", use_container_width=True)

    # Run segmentation when button clicked
    if run_clicked:
    
        img_np = np.array(image2)
        results = model.predict(img_np, verbose=False)[0]
        seg_img = results.plot()

        # Display segmentation result beside original
        seg_img_placeholder.image(seg_img, caption="Segmentation Result", use_container_width=True)

        # Convert segmentation result to bytes for download
        seg_pil = Image.fromarray(seg_img)
        buf = io.BytesIO()
        seg_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download button
        st.download_button(
            label="📥 Download Segmentation Result",
            data=byte_im,
            file_name="lung_nodule_segmentation.png",
            mime="image/png",
            use_container_width=True
        )