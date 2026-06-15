import os
import nibabel as nib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    BASE_DIR,
    "DATA",
    "RAW",
    "brats2020",
    "BraTS2020_TrainingData",
    "MICCAI_BraTS2020_TrainingData",
    "BraTS20_Training_001",
    "BraTS20_Training_001_flair.nii"
)

img = nib.load(file_path)
data = img.get_fdata()

print("Loaded file:", file_path)
print("Shape:", data.shape)

slice_index = data.shape[2] // 2

plt.figure(figsize=(6, 6))
plt.imshow(data[:, :, slice_index], cmap="gray")
plt.title(f"BraTS FLAIR Slice {slice_index}")
plt.axis("off")
plt.show()