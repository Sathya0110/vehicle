import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Vehicle Detection using YOLOv8")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    annotated = results[0].plot()

    st.image(annotated, caption="Detected Vehicles")
