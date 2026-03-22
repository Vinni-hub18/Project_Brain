import nibabel as nib
import matplotlib.pyplot as plt

# Correct path + correct filename
img = nib.load('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii')

data = img.get_fdata()

print("Shape:", data.shape)

plt.imshow(data[:, :, 80], cmap='gray')
plt.title("MRI Slice (FLAIR)")
plt.axis('off')
plt.show()