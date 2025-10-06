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
col1, col2 = st.columns([1, 3])
with col1:
    st.image(image)
with col2:
    st.title("i3L AI-based Lung Nodule Segmentation")

st.subheader("Lung Nodule Segmentation")

uploaded_file = st.file_uploader("Upload a Lung Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image2 = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image2)

    if st.button("Run Segmentation"):
        # Run YOLO segmentation
        results = model.predict(img_np)[0]
        seg_img = results.plot()  # YOLO overlay result
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        # Transparency control
        alpha = st.slider("Adjust Segmentation Transparency", 0.0, 1.0, 0.5)

        # Blend the original and segmentation overlay
        blended = cv2.addWeighted(img_np, 1 - alpha, seg_img, alpha, 0)

        # --- Show side-by-side images ---
        colA, colB = st.columns(2)
        with colA:
            st.image(img_np, caption="Original Image", use_container_width=True)
        with colB:
            st.image(blended, caption="Segmentation Overlay", use_container_width=True)

        # --- Create side-by-side composite for download ---
        # Ensure both have same height
        h = min(img_np.shape[0], blended.shape[0])
        img_np_resized = cv2.resize(img_np, (int(img_np.shape[1] * h / img_np.shape[0]), h))
        blended_resized = cv2.resize(blended, (int(blended.shape[1] * h / blended.shape[0]), h))
        combined = np.concatenate((img_np_resized, blended_resized), axis=1)

        # Convert to PIL for download
        combined_pil = Image.fromarray(combined)
        buf = io.BytesIO()
        combined_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download button
        st.download_button(
            label="Download Side-by-Side Result",
            data=byte_im,
            file_name="lung_nodule_segmentation_result.png",
            mime="image/png"
        )
