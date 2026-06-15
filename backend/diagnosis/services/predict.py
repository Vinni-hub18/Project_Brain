import os
import cv2
import numpy as np
import torch
from PIL import Image

from .preprocess import preprocess_image, PreprocessError
from .gradcam import create_overlay
from model.unet import UNet


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
MEDIA_ROOT = os.path.join(BASE_DIR, "backend", "media")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()


class PredictionError(Exception):
    pass


def remove_small_components(mask_uint8, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned = np.zeros_like(mask_uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def analyze_mask(prob_map_np, cleaned_mask_uint8):
    mask_bool = cleaned_mask_uint8 > 0
    mask_area_pixels = int(mask_bool.sum())
    total_pixels = int(mask_bool.size)
    mask_area_ratio = float(mask_area_pixels / total_pixels)

    max_probability = float(prob_map_np.max())
    mean_mask_probability = float(prob_map_np[mask_bool].mean()) if mask_area_pixels > 0 else 0.0

    return {
        "mask_area_pixels": mask_area_pixels,
        "mask_area_ratio": mask_area_ratio,
        "max_probability": max_probability,
        "mean_mask_probability": mean_mask_probability,
    }


def decide_prediction(preprocess_stats, mask_stats):
    foreground_ratio = preprocess_stats["brain_likelihood"]["foreground_ratio"]
    center_border_contrast = preprocess_stats["brain_likelihood"]["center_border_contrast"]

    if foreground_ratio < 0.05 or center_border_contrast < 0.02:
        return {
            "prediction": "invalid_input",
            "confidence_score": 0.0,
            "insight_text": "Uploaded image does not appear to be a valid brain MRI slice for this model.",
            "validation_message": "Input rejected by domain validation."
        }

    if mask_stats["mask_area_pixels"] == 0:
        confidence = round((1.0 - mask_stats["max_probability"]) * 100, 2)
        return {
            "prediction": "no_tumor",
            "confidence_score": confidence,
            "insight_text": "No significant tumor-like region detected after post-processing.",
            "validation_message": None
        }

    if mask_stats["mask_area_ratio"] < 0.003:
        confidence = round(mask_stats["mean_mask_probability"] * 100, 2)
        return {
            "prediction": "uncertain",
            "confidence_score": confidence,
            "insight_text": "A very small suspicious region was detected, but it is below the reliable decision threshold.",
            "validation_message": "Predicted region too small for confident decision."
        }

    confidence = round(mask_stats["mean_mask_probability"] * 100, 2)
    return {
        "prediction": "tumor_suspected",
        "confidence_score": confidence,
        "insight_text": "Post-processed segmentation identified a suspicious region consistent with possible tumor tissue. Clinical confirmation is required.",
        "validation_message": None
    }


def predict_mask(file_path, scan_id, threshold=0.5, min_component_area=500):
    try:
        prep = preprocess_image(file_path, target_size=(224, 224))
    except PreprocessError as e:
        return {
            "ok": False,
            "prediction": "invalid_input",
            "confidence_score": 0.0,
            "insight_text": "Uploaded image could not be analyzed reliably.",
            "validation_message": str(e),
            "mask_area_pixels": 0,
            "mask_area_ratio": 0.0,
            "max_probability": 0.0,
            "mean_mask_probability": 0.0,
            "mask_path": None,
            "overlay_path": None,
        }

    image_norm = prep["image_norm"]

    input_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        raw_mask = (prob > threshold).float()

    raw_mask_np = raw_mask.squeeze().cpu().numpy()
    prob_map_np = prob.squeeze().cpu().numpy()
    raw_mask_uint8 = (raw_mask_np * 255).astype(np.uint8)

    cleaned_mask_uint8 = remove_small_components(raw_mask_uint8, min_area=min_component_area)

    mask_stats = analyze_mask(prob_map_np, cleaned_mask_uint8)
    decision = decide_prediction(prep, mask_stats)

    mask_dir = os.path.join(MEDIA_ROOT, "scans", "masks")
    heatmap_dir = os.path.join(MEDIA_ROOT, "scans", "heatmaps")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    mask_name = f"mask_scan_{scan_id}.png"
    overlay_name = f"overlay_scan_{scan_id}.png"
    mask_path = os.path.join(mask_dir, mask_name)
    overlay_path = os.path.join(heatmap_dir, overlay_name)

    Image.fromarray(cleaned_mask_uint8).save(mask_path)
    create_overlay(image_norm, cleaned_mask_uint8 / 255.0, overlay_path)

    return {
        "ok": True,
        "prediction": decision["prediction"],
        "confidence_score": decision["confidence_score"],
        "insight_text": decision["insight_text"],
        "validation_message": decision["validation_message"],
        "mask_area_pixels": mask_stats["mask_area_pixels"],
        "mask_area_ratio": mask_stats["mask_area_ratio"],
        "max_probability": round(mask_stats["max_probability"] * 100, 2),
        "mean_mask_probability": round(mask_stats["mean_mask_probability"] * 100, 2),
        "mask_path": mask_path,
        "overlay_path": overlay_path,
    }