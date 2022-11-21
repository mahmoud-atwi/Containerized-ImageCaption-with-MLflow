import streamlit as st
import requests
import pandas as pd
import io
import json
from io import BytesIO
from PIL import Image

st.title('Image Caption')

# Set FastAPI endpoint
endpoint = 'http://host.docker.internal:8000/predict'

def load_image(image_file):
	img = Image.open(image_file)
	return img

uploaded_image = st.file_uploader(
    '', type=['jpeg'], accept_multiple_files=False)

if uploaded_image is not None:
    # To read file as bytes:
    st.image(load_image(uploaded_image), width=704)

    bytes_data = uploaded_image.getvalue()

    files = {"file": ('media', bytes_data, "multipart/form-data")}

    with st.spinner('Prediction in Progress. Please Wait...'):
        prediction = requests.post(endpoint, files=files, timeout=8000)

    data=json.dumps(prediction.json())

    if data == '"Fail"':
        st.error("not enough image data, try another one", icon="🚨")
    else:
        st.success('Caption prediction generated successfully')
        st.write(data)