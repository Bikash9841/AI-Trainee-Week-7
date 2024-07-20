import streamlit as st
from api import app
import requests
from PIL import Image
import uvicorn
import subprocess
import threading
import io


st.title("Predict What the Image is!")

col1, col2 = st.columns(2)

# create placeholder to display the selected image
ph0 = col1.empty()


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)


def start_fastapi():
    # Check if the server is already running by looking for the port
    try:
        subprocess.check_call(["lsof", "-i", ":8000"])
        print("FastAPI server is already running.")
    except subprocess.CalledProcessError:
        print("Starting FastAPI server...")
        fastapi_thread = threading.Thread(target=run_fastapi)
        fastapi_thread.daemon = True
        fastapi_thread.start()


def main():

    # Select the picture
    file = col2.file_uploader("Choose a image file")
    if file is not None:
        img = Image.open(file)
        ph0.write(img)
        image_bytes = io.BytesIO()
        img.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()

    # predict button
    if st.button("Predict"):
        # Store the HTML block as a string
        response = requests.post(
            "http://0.0.0.0:8000/predict/", files={'img_file': ('pic.jpg', image_bytes, 'image/jpeg')})

        st.success("Process Complete")
        col2.title(f"Prediction: {(response.json())['label']}")


if __name__ == '__main__':

    start_fastapi()

    main()
