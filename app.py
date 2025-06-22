# app.py
import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import Generator
from utils import generate_digit_images

# Constants
noise_dim = 100
model_path = "generator.pth"

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(noise_dim=noise_dim).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# UI
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    with st.spinner("Generating images..."):
        images = generate_digit_images(generator, digit, num_images=5)
    
    st.write(f"Generated images of digit {digit}:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        fig, ax = plt.subplots()
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.axis("off")
        col.pyplot(fig)
