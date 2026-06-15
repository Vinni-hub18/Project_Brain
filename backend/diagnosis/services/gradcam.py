import os
import cv2
import numpy as np


def create_overlay(image_norm, pred_mask, output_path):
    image_norm = np.clip(image_norm, 0, 1)
    pred_mask = np.clip(pred_mask, 0, 1)

    image_rgb = np.uint8(255 * image_norm)
    image_rgb = np.stack([image_rgb] * 3, axis=-1)

    mask_uint8 = np.uint8(pred_mask * 255)
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb, 0.65, heatmap, 0.35, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return output_path