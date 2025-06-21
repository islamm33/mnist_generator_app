import streamlit as st
import torch
import numpy as np
from PIL import Image

# Add to top of app.py
import time
from datetime import datetime

# Keep-alive ping
def keep_alive():
    while True:
        time.sleep(300)  # Ping every 5 minutes
        print(f"Heartbeat: {datetime.now()}")

import threading
threading.Thread(target=keep_alive, daemon=True).start()

# 1. Load Trained Generator
@st.cache_resource  # Cache to avoid reloading
def load_model():
    # Define the generator architecture (must match training)
    generator = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(256, 512),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(512, 784),
        torch.nn.Tanh()
    )
    # Load pre-trained weights
    generator.load_state_dict(torch.load("mnist_generator.pth", map_location='cpu'))
    generator.eval()
    return generator

# 2. App Interface
st.title("MNIST Digit Generator")
digit = st.selectbox("Select Digit", list(range(10)))  # 0-9

if st.button("Generate"):
    generator = load_model()
    with torch.no_grad():
        # Generate 5 images from random noise
        noise = torch.randn(5, 100)
        generated = generator(noise).numpy()
    
    # Display images
    cols = st.columns(5)
    for i in range(5):
        img = (generated[i].squeeze() * 127.5 + 127.5).astype(np.uint8)  # Convert to 0-255
        cols[i].image(Image.fromarray(img), caption=f"Digit {digit}")
