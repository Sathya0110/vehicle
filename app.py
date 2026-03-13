import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚗 Vehicle Detection using YOLOv8")

st.write("Upload an image containing vehicles.")

# Load model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting vehicles...")

    # Detect only vehicle classes
    results = model(img, classes=[2,3,5,7])

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Vehicles", channels="BGR")

    if len(results[0].boxes) == 0:
        st.warning("No vehicles detected in this image.")
