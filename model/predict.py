import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet import UNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "DATA", "processed", "images")
MASK_PATH = os.path.join(BASE_DIR, "DATA", "processed", "masks")
OUTPUT_PATH = os.path.join(BASE_DIR, "model", "prediction_overlay.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

sample_file = sorted(os.listdir(IMAGE_PATH))[0]
image = np.load(os.path.join(IMAGE_PATH, sample_file)).astype(np.float32)
mask = np.load(os.path.join(MASK_PATH, sample_file)).astype(np.float32)

input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output)
    pred_mask = (prob > 0.5).float()

image_np = image
pred_np = pred_mask.squeeze().cpu().numpy()
mask_np = mask

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(image_np, cmap="gray")
axs[0].set_title("Input Image")
axs[0].axis("off")

axs[1].imshow(image_np, cmap="gray")
axs[1].imshow(mask_np, cmap="jet", alpha=0.4)
axs[1].set_title("Ground Truth")
axs[1].axis("off")

axs[2].imshow(image_np, cmap="gray")
axs[2].imshow(pred_np, cmap="jet", alpha=0.4)
axs[2].set_title("Prediction")
axs[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()

print(f"Prediction saved to: {OUTPUT_PATH}")