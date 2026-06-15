import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet import UNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "DATA", "processed", "images")
OUTPUT_PATH = os.path.join(BASE_DIR, "model", "gradcam_conv4_overlay.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

sample_file = sorted(os.listdir(IMAGE_PATH))[0]
image = np.load(os.path.join(IMAGE_PATH, sample_file)).astype(np.float32)

image_min = image.min()
image_max = image.max()
image_norm = (image - image_min) / (image_max - image_min + 1e-8)

input_tensor = torch.tensor(image_norm).unsqueeze(0).unsqueeze(0).to(device)

activations = []
gradients = []

target_layer = model.conv4.block[3]

def forward_hook(module, inp, out):
    activations.clear()
    activations.append(out)

def backward_hook(module, grad_in, grad_out):
    gradients.clear()
    gradients.append(grad_out[0])

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

output = model(input_tensor)
prob = torch.sigmoid(output)

pred_mask = (prob > 0.5).float()
target_score = (prob * pred_mask).sum()

if target_score.item() == 0:
    target_score = prob.sum()

model.zero_grad()
target_score.backward()

acts = activations[0].detach()
grads = gradients[0].detach()

weights = grads.mean(dim=(2, 3), keepdim=True)
cam = (weights * acts).sum(dim=1, keepdim=True)
cam = F.relu(cam)
cam = F.interpolate(cam, size=image.shape, mode="bilinear", align_corners=False)

cam = cam.squeeze().cpu().numpy()
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)

pred_mask_np = pred_mask.squeeze().detach().cpu().numpy()

heatmap = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

image_rgb = np.uint8(255 * image_norm)
image_rgb = np.stack([image_rgb] * 3, axis=-1)

overlay = cv2.addWeighted(image_rgb, 0.65, heatmap, 0.35, 0)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(image_norm, cmap="gray")
axs[0].set_title("Input MRI")
axs[0].axis("off")

axs[1].imshow(pred_mask_np, cmap="gray")
axs[1].set_title("Predicted Mask")
axs[1].axis("off")

axs[2].imshow(cam, cmap="jet")
axs[2].set_title("Grad-CAM Heatmap (conv4)")
axs[2].axis("off")

axs[3].imshow(overlay)
axs[3].set_title("Grad-CAM Overlay (conv4)")
axs[3].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()

forward_handle.remove()
backward_handle.remove()

print(f"Saved conv4 Grad-CAM to: {OUTPUT_PATH}")