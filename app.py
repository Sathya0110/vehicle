import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚦 AI Traffic Monitoring System")

st.write("Upload a road image to detect and count vehicles.")

# Load better accuracy model
model = YOLO("yolov8m.pt")

uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect vehicles with confidence filtering
    results = model(img, classes=[2,3,5,7], conf=0.5)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Vehicles", channels="BGR")

    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy() if boxes is not None else []

    car = 0
    motorcycle = 0
    bus = 0
    truck = 0

    for c in classes:
        if int(c) == 2:
            car += 1
        elif int(c) == 3:
            motorcycle += 1
        elif int(c) == 5:
            bus += 1
        elif int(c) == 7:
            truck += 1

    total = car + motorcycle + bus + truck

    st.subheader("🚗 Vehicle Count")

    st.write(f"Cars: {car}")
    st.write(f"Motorcycles: {motorcycle}")
    st.write(f"Buses: {bus}")
    st.write(f"Trucks: {truck}")

    st.subheader(f"🚘 Total Vehicles: {total}")

    # Traffic level estimation
    if total <= 3:
        st.success("Traffic Level: LOW")
    elif total <= 7:
        st.warning("Traffic Level: MEDIUM")
    else:
        st.error("Traffic Level: HIGH")
