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

uploaded_file = st.file_uploader("Upload a Chest Image", type=['png', 'jpg', 'jpeg'])

# Keep state for reusing segmentation result
if "seg_mask" not in st.session_state:
    st.session_state.seg_mask = None
    st.session_state.orig_image = None

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")

    st.markdown("<hr>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_clicked = st.button("🔍 Run Segmentation", use_container_width=True)

    if run_clicked:
        with st.spinner("Running segmentation..."):
            img_np = np.array(image2)
            results = model.predict(img_np, verbose=False)[0]

            # Create an empty mask (same size as input)
            mask = np.zeros_like(img_np, dtype=np.uint8)

            # Loop through detected masks
            if results.masks is not None:
                for m in results.masks.data:
                    m = m.cpu().numpy()
                    m = (m * 255).astype(np.uint8)
                    m = cv2.resize(m, (img_np.shape[1], img_np.shape[0]))
                    mask[m > 128] = (0, 255, 0)  # green overlay for segmentation

            st.session_state.seg_mask = mask
            st.session_state.orig_image = img_np

        st.success("✅ Segmentation complete! Adjust transparency below.")

    # If segmentation mask exists
    if st.session_state.seg_mask is not None:
        st.markdown("### 🩶 Adjust Segmentation Transparency")
        alpha = st.slider("Transparency", 0.0, 1.0, 0.5, 0.05)

        # Blend original and mask (no text or confidence shown)
        blended = cv2.addWeighted(
            st.session_state.orig_image, 1.0,
            st.session_state.seg_mask, alpha, 0
        )

        st.image(blended, caption=f"Segmentation Result (Transparency: {alpha:.2f})", use_container_width=True)

        # Prepare image for download
        seg_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        seg_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Segmentation Result",
            data=byte_im,
            file_name="lung_nodule_segmentation.png",
            mime="image/png",
            use_container_width=True
        )
