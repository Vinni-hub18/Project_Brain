import os
import glob
import nibabel as nib
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(
    BASE_DIR,
    "DATA",
    "RAW",
    "brats2020",
    "BraTS2020_TrainingData",
    "MICCAI_BraTS2020_TrainingData"
)

OUT_IMG_DIR = os.path.join(BASE_DIR, "DATA", "processed", "images")
OUT_MASK_DIR = os.path.join(BASE_DIR, "DATA", "processed", "masks")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


def normalize_slice(img_slice):
    img_slice = img_slice.astype(np.float32)
    img_slice = img_slice - img_slice.min()
    max_val = img_slice.max()
    if max_val > 0:
        img_slice = img_slice / max_val
    return img_slice


def save_preview(image_slice, mask_slice, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(mask_slice, cmap="jet", alpha=0.35)
    plt.axis("off")
    plt.title("Processed Sample Slice")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def process_case(case_dir):
    case_id = os.path.basename(case_dir)

    flair_path = os.path.join(case_dir, f"{case_id}_flair.nii")
    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii")

    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        print(f"Skipping {case_id} - flair or seg file missing")
        return 0

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    saved_count = 0

    for slice_idx in range(flair.shape[2]):
        image_slice = flair[:, :, slice_idx]
        mask_slice = seg[:, :, slice_idx]

        if np.max(mask_slice) == 0:
            continue

        image_slice = normalize_slice(image_slice)

        image_resized = resize(
            image_slice,
            (224, 224),
            preserve_range=True,
            anti_aliasing=True
        )

        mask_resized = resize(
            mask_slice,
            (224, 224),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        image_resized = image_resized.astype(np.float32)
        mask_resized = mask_resized.astype(np.uint8)

        image_name = f"{case_id}_{slice_idx:03d}.npy"
        mask_name = f"{case_id}_{slice_idx:03d}.npy"

        np.save(os.path.join(OUT_IMG_DIR, image_name), image_resized)
        np.save(os.path.join(OUT_MASK_DIR, mask_name), mask_resized)

        saved_count += 1

    return saved_count


def main():
    case_dirs = sorted([
        folder for folder in glob.glob(os.path.join(RAW_DIR, "BraTS20_Training_*"))
        if os.path.isdir(folder)
    ])

    total_saved = 0

    print(f"Found {len(case_dirs)} cases")

    for i, case_dir in enumerate(case_dirs, start=1):
        count = process_case(case_dir)
        total_saved += count
        print(f"[{i}/{len(case_dirs)}] {os.path.basename(case_dir)} -> saved {count} slices")

    print(f"\nTotal saved slices: {total_saved}")

    sample_images = sorted(glob.glob(os.path.join(OUT_IMG_DIR, "*.npy")))
    sample_masks = sorted(glob.glob(os.path.join(OUT_MASK_DIR, "*.npy")))

    if sample_images and sample_masks:
        sample_image = np.load(sample_images[0])
        sample_mask = np.load(sample_masks[0])

        preview_path = os.path.join(BASE_DIR, "DATA", "processed", "sample_preview.png")
        save_preview(sample_image, sample_mask, preview_path)
        print(f"Preview saved at: {preview_path}")


if __name__ == "__main__":
    main()