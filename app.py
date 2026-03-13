import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚦 AI Traffic Monitoring System")

st.write("Upload a road image to detect and count vehicles.")

# Load YOLO model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Vehicle classes
    vehicle_classes = [2, 3, 5, 7]

    results = model(img, classes=vehicle_classes)

    annotated = results[0].plot()

    boxes = results[0].boxes

    vehicle_count = len(boxes)

    st.image(annotated, caption="Detected Vehicles", channels="BGR")

    st.subheader(f"🚗 Total Vehicles Detected: {vehicle_count}")

    # Traffic density estimation
    if vehicle_count <= 3:
        st.success("Traffic Level: LOW")
    elif vehicle_count <= 7:
        st.warning("Traffic Level: MEDIUM")
    else:
        st.error("Traffic Level: HIGH")
