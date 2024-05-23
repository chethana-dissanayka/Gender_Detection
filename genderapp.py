import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Function to perform gender detection on a single image
def detect_gender(image):
    # Apply face detection
    faces, confidences = cv.detect_face(image)

    for face, confidence in zip(faces, confidences):
        # Get corner points of face rectangle
        startX, startY, endX, endY = face[0], face[1], face[2], face[3]

        # Crop the detected face region
        face_crop = image[startY:endY, startX:endX]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection model
        conf = gender_model.predict(face_crop)[0]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = gender_classes[idx]

        accuracy = f"{conf[idx]*100:.2f}%"

        return label, accuracy

# Streamlit app
st.header('Gender Detection CNN Model')

# Load the pre-trained gender detection model
gender_model = load_model('gender_detection.keras')
gender_classes = ['man', 'woman']

# File upload for images
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform gender detection on the image
    label, accuracy = detect_gender(image)
    
    # Display the image with the desired size
    st.image(image, channels="BGR", width=300)
     
  
    st.write(f"<span style='font-size: x-large; color: yellow;'>This is  a : {label} -{accuracy}.</span>", unsafe_allow_html=True)
