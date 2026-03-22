import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

base_path = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

os.makedirs('dataset/images', exist_ok=True)
os.makedirs('dataset/labels', exist_ok=True)

count = 0

# Loop through ALL patient folders
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    flair_path = os.path.join(folder_path, f"{folder}_flair.nii")
    seg_path = os.path.join(folder_path, f"{folder}_seg.nii")

    # Check if files exist
    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        continue

    img = nib.load(flair_path)
    mask = nib.load(seg_path)

    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # Loop through slices
    for i in range(img_data.shape[2]):
        slice_img = img_data[:, :, i]
        slice_mask = mask_data[:, :, i]

        if np.max(slice_img) == 0:
            continue

        # Save image
        plt.imsave(f'dataset/images/img_{count}.png', slice_img, cmap='gray')

        # Label
        if np.max(slice_mask) > 0:
            label = 1
        else:
            label = 0

        with open('dataset/labels/labels.txt', 'a') as f:
            f.write(f'img_{count}.png {label}\n')

        count += 1

print("Total images created:", count)