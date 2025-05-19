import streamlit as st
import easyocr
import numpy as np
from PIL import Image

# load once
reader = easyocr.Reader(['en'], gpu=False)

st.title("EasyOCR on localhost")
uploaded = st.file_uploader("Choose an image…", type=["png","jpg","jpeg"])
if uploaded:
    img = np.array(Image.open(uploaded))
    st.image(img, caption="Uploaded image", use_column_width=True)
    st.write("Running OCR…")
    for bbox, text, prob in reader.readtext(img):
        st.write(f"**{text}** (confidence {prob:.2f})")
