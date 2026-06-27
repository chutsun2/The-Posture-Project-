
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import depth_pro
import torch.nn.functional as F

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
# Load image
image, _, f_px = depth_pro.load_rgb(r"C:\Users\ChuTs\OneDrive - Government of Ontario\Desktop\Current Projects\The-Posture-Project-\ml-depth-pro\data\maltese-portrait.jpg")

# Transform + send to GPU
image = transform(image)
image = image.unsqueeze(0)
image = F.interpolate(
    image,
    size=(1536, 1536),
    mode=interpolation_mode,
    align_corners=False,
)
torch.onnx.export(model, image, r"C:\Users\ChuTs\OneDrive - Government of Ontario\Desktop\Current Projects\The-Posture-Project-\model.onnx")

import os
print(os.getcwd())

