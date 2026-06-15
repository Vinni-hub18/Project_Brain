import os
import nibabel as nib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

case_dir = os.path.join(
    BASE_DIR,
    "DATA",
    "RAW",
    "brats2020",
    "BraTS2020_TrainingData",
    "MICCAI_BraTS2020_TrainingData",
    "BraTS20_Training_001"
)

flair_path = os.path.join(case_dir, "BraTS20_Training_001_flair.nii")
mask_path = os.path.join(case_dir, "BraTS20_Training_001_seg.nii")

flair_img = nib.load(flair_path)
mask_img = nib.load(mask_path)

flair = flair_img.get_fdata()
mask = mask_img.get_fdata()

slice_idx = flair.shape[2] // 2

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(flair[:, :, slice_idx], cmap="gray")
axs[0].set_title("FLAIR Slice")
axs[0].axis("off")

axs[1].imshow(flair[:, :, slice_idx], cmap="gray")
axs[1].imshow(mask[:, :, slice_idx], cmap="jet", alpha=0.5)
axs[1].set_title("FLAIR + Mask")
axs[1].axis("off")

plt.tight_layout()
plt.show()