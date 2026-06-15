import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import pydicom
from pydicom.pixels import apply_rescale


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}


class PreprocessError(Exception):
    pass


def normalize_minmax(image_np):
    image_np = image_np.astype(np.float32)
    min_val = float(image_np.min())
    max_val = float(image_np.max())

    if max_val - min_val < 1e-8:
        raise PreprocessError("Image has near-constant intensity and cannot be analyzed.")

    return (image_np - min_val) / (max_val - min_val)


def center_crop_to_square(image_np):
    h, w = image_np.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image_np[y0:y0 + side, x0:x0 + side]


def validate_basic_image_quality(image_np):
    if image_np is None or image_np.size == 0:
        raise PreprocessError("Image is empty or unreadable.")

    if image_np.ndim != 2:
        raise PreprocessError("Only single-channel grayscale images are supported.")

    h, w = image_np.shape
    if h < 64 or w < 64:
        raise PreprocessError("Image resolution is too small for reliable analysis.")

    aspect_ratio = max(h, w) / max(1, min(h, w))
    if aspect_ratio > 2.0:
        raise PreprocessError("Image aspect ratio is not suitable for brain scan analysis.")

    std_val = float(np.std(image_np))
    if std_val < 5:
        raise PreprocessError("Image has too little intensity variation for analysis.")


def estimate_brain_likelihood(image_np_norm):
    h, w = image_np_norm.shape
    center = image_np_norm[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    border_mask = np.ones_like(image_np_norm, dtype=bool)
    border_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = False
    border = image_np_norm[border_mask]

    center_mean = float(center.mean())
    border_mean = float(border.mean())
    contrast = center_mean - border_mean

    foreground_ratio = float((image_np_norm > 0.12).mean())
    return {
        "center_mean": center_mean,
        "border_mean": border_mean,
        "center_border_contrast": contrast,
        "foreground_ratio": foreground_ratio,
    }


def validate_brain_like_structure(image_np_norm):
    stats = estimate_brain_likelihood(image_np_norm)

    if stats["foreground_ratio"] < 0.05:
        raise PreprocessError("Image appears mostly empty or background only.")

    if stats["center_border_contrast"] < 0.02:
        raise PreprocessError("Image does not resemble a centered brain scan slice.")

    return stats


def load_standard_image(file_path):
    try:
        image = Image.open(file_path).convert("L")
    except UnidentifiedImageError:
        raise PreprocessError("Uploaded image file could not be decoded.")

    image_np = np.array(image, dtype=np.float32)
    return image_np


def load_dicom_image(file_path):
    try:
        ds = pydicom.dcmread(file_path)
        image_np = ds.pixel_array.astype(np.float32)
        image_np = apply_rescale(image_np, ds).astype(np.float32)
    except Exception as e:
        raise PreprocessError(f"Failed to read DICOM image: {str(e)}")

    if image_np.ndim == 3:
        image_np = image_np[..., 0]

    return image_np


def preprocess_image(file_path, target_size=(224, 224)):
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise PreprocessError(f"Unsupported file format: {ext}")

    if ext == ".dcm":
        image_np = load_dicom_image(file_path)
    else:
        image_np = load_standard_image(file_path)

    validate_basic_image_quality(image_np)

    image_np = center_crop_to_square(image_np)

    pil_image = Image.fromarray(image_np).convert("L")
    pil_image = pil_image.resize(target_size, Image.Resampling.BILINEAR)
    image_resized = np.array(pil_image, dtype=np.float32)

    image_norm = normalize_minmax(image_resized)
    stats = validate_brain_like_structure(image_norm)

    return {
        "image_norm": image_norm,
        "input_height": int(image_np.shape[0]),
        "input_width": int(image_np.shape[1]),
        "brain_likelihood": stats,
    }