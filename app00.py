import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile

# Load YOLOv8 segmentation model

model = YOLO('best_seg_LungNodule.pt')

# Streamlit UI
st.title("🎯 YOLOv8 Segmentation with Streamlit")

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

        # Optional: show object classes
        boxes = results.boxes
        masks = results.masks

        st.write("Detected Classes:", 
            [model.names[int(cls)] for cls in boxes.cls])

        if masks is not None:
            st.write(f"Number of masks: {len(masks.data)}")
