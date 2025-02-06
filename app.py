import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Define model structure
input_size = 784
hidden_size = [128, 64]
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_size[0]),
    nn.ReLU(),
    nn.Linear(hidden_size[0], hidden_size[1]),
    nn.ReLU(),
    nn.Linear(hidden_size[1], output_size),
    nn.LogSoftmax(dim=1)
)

model.load_state_dict(torch.load(r"D:\SEM_6\Deep Learning NN\LAB\handwritten_model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("Handwritten Digit Recognition via Webcam 🖊️")
st.write("Use your webcam to draw a digit (0-9), and the model will predict it!")

# Capture webcam feed
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Open the uploaded image (from webcam capture)
    image = Image.open(camera_input)
    st.image(image, caption="Captured Image", use_container_width=True)

    # Preprocess the image
    img = transform(image)
    img = img.view(1, 784)  # Flatten the image

    # Predict the digit
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)  # Convert log probabilities to probabilities
    prediction = torch.argmax(ps, dim=1).item()

    # Display the predicted digit
    st.write(f"🧠 **Predicted Digit: {prediction}** 🎯")

<a href="https://dl.google.com/android/repository/platform-tools-latest-windows.zip" rel="noopener noreferrer" target="_blank">Android SDK Platform Tools ZIP file for Windows</a>