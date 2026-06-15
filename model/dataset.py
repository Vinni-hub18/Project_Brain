import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    def __init__(self, base_dir, split_file):
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, "DATA", "processed", "images")
        self.mask_dir = os.path.join(base_dir, "DATA", "processed", "masks")

        with open(split_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)

        mask = (mask > 0).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask