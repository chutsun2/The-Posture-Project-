
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import depth_pro

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load image
image, _, f_px = depth_pro.load_rgb(r"C:\Users\ChuTs\OneDrive - Government of Ontario\Desktop\Current Projects\The-Posture-Project-\ml-depth-pro\data\example.jpg")

# Transform + send to GPU
image = transform(image)

# Inference
with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

depth = prediction["depth"]

print(depth)
