import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("brain_tumor_model.h5")

# Load image (change this path)
img_path = "dataset/images/img_6.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))

# Normalize
img = img / 255.0

# Reshape for model
img = img.reshape(1, 128, 128, 1)

# Predict
prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Tumor Detected 😶")
else:
    print("No Tumor 😊")