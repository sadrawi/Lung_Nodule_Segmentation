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
image = Image.open('i3LUniversity.png')
col1, col2 = st.columns([1, 3])
with col1:
    st.image(image, use_container_width=True)
with col2:
    st.title("i3L AI-based Lung Nodule Segmentation")

st.header("Lung Nodule Segmentation")

# File upload
uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

# Store results in session state
if "seg_result" not in st.session_state:
    st.session_state.seg_result = None
    st.session_state.orig_image = None

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")
    st.session_state.orig_image = np.array(image2)

    # Create side-by-side columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.orig_image, 
                 caption="Original Image", 
                 use_container_width=True)
    with col2:
        seg_img_placeholder = st.empty()
        if st.session_state.seg_result is None:
            placeholder_img = np.ones((200, 200, 3), dtype=np.uint8) * 230
            seg_img_placeholder.image(
                placeholder_img, 
                caption="Segmentation result will appear here after running the model.", 
                use_container_width=True
            )

    # Run segmentation button
    st.markdown("<hr>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_clicked = st.button("Run Segmentation", 
                                use_container_width=True)

    # Run YOLO segmentation
    if run_clicked:
        with st.spinner("Running segmentation..."):
            results = model.predict(st.session_state.orig_image, verbose=False)[0]
            seg_img = results.plot()
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        st.session_state.seg_result = seg_img
        st.success("✅ Segmentation complete! Adjust transparency below.")

    # Show transparency slider and side-by-side results
    if st.session_state.seg_result is not None:
        alpha = st.slider("Transparency", 0.0, 1.0, 0.5, 0.05)

        blended = cv2.addWeighted(
            st.session_state.orig_image, 1 - alpha,
            st.session_state.seg_result, alpha, 0
        )

        # Update the right column with blended image
        seg_img_placeholder.image(
            blended,
            caption=f"Segmentation Result (Transparency: {alpha:.2f})",
            use_container_width=True
        )

        # Create a separator line (vertical)
        h = st.session_state.orig_image.shape[0]
        separator = np.ones((h, 10, 3), dtype=np.uint8) * 128  # white line (10px wide)

        # Combine side by side with separator
        combined = np.hstack((st.session_state.orig_image, separator, blended))

        # Convert to PIL for download
        combined_pil = Image.fromarray(combined)
        buf = io.BytesIO()
        combined_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download button
        st.download_button(
            label="Download Side-by-Side Result",
            data=byte_im,
            file_name="lung_nodule_segmentation.png",
            mime="image/png",
            use_container_width=True
        )
