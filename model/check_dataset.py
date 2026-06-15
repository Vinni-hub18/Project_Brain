import os
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SPLIT = os.path.join(BASE_DIR, "DATA", "splits", "train.txt")

dataset = BrainTumorDataset(BASE_DIR, TRAIN_SPLIT)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, masks = next(iter(loader))

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)
print("Mask unique values:", masks.unique())