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

# File upload
uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

# Initialize session state
if "seg_result" not in st.session_state:
    st.session_state.seg_result = None
    st.session_state.orig_image = None

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")
    st.session_state.orig_image = np.array(image2)

    # Columns for original and segmented views
    colA, colB = st.columns(2)

    # Show the original image immediately (left)
    with colA:
        st.image(image2, caption="Original Image", use_container_width=True)

    # Segmentation column (right)
    with colB:
        if st.session_state.seg_result is not None:
            st.image(st.session_state.seg_result, caption="Segmented Result", use_container_width=True)
        else:
            st.info("Segmentation result will appear here after running the model.")

    # Centered Run Segmentation button
    st.markdown("<hr>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_clicked = st.button("🔍 Run Segmentation", use_container_width=True)

    # Run segmentation when button pressed
    if run_clicked:
        with st.spinner("Running segmentation..."):
            img_np = np.array(image2)
            results = model.predict(img_np, verbose=False)[0]
            seg_img = results.plot()
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        st.session_state.seg_result = seg_img
        st.success("✅ Segmentation complete! Adjust transparency below.")

# If segmentation result exists, show transparency control and download
if st.session_state.seg_result is not None:
    st.markdown("### 🩶 Adjust Segmentation Transparency")
    alpha = st.slider("Transparency", 0.0, 1.0, 0.5, 0.05)

    blended = cv2.addWeighted(
        st.session_state.orig_image, 1 - alpha,
        st.session_state.seg_result, alpha, 0
    )

    colA, colB = st.columns(2)
    with colA:
        st.image(st.session_state.orig_image, caption="Original Image", use_container_width=True)
    with colB:
        st.image(blended, caption=f"Segmentation Overlay (α={alpha:.2f})", use_container_width=True)

    # Prepare for download
    h = min(st.session_state.orig_image.shape[0], blended.shape[0])
    orig_resized = cv2.resize(st.session_state.orig_image, (int(st.session_state.orig_image.shape[1] * h / st.session_state.orig_image.shape[0]), h))
    blended_resized = cv2.resize(blended, (int(blended.shape[1] * h / blended.shape[0]), h))
    separator = 255 * np.ones((h, 10, 3), dtype=np.uint8)
    combined = np.concatenate((orig_resized, separator, blended_resized), axis=1)

    combined_pil = Image.fromarray(combined)
    buf = io.BytesIO()
    combined_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Side-by-Side Result",
        data=byte_im,
        file_name="lung_nodule_segmentation_side_by_side.png",
        mime="image/png",
        use_container_width=True
    )
