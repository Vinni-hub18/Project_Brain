import os
import glob
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_DIR = os.path.join(BASE_DIR, "DATA", "processed", "images")
MASK_DIR = os.path.join(BASE_DIR, "DATA", "processed", "masks")
SPLIT_DIR = os.path.join(BASE_DIR, "DATA", "splits")

os.makedirs(SPLIT_DIR, exist_ok=True)

random.seed(42)

image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.npy")))

# Get unique patient IDs
patient_to_files = {}

for img_path in image_files:
    filename = os.path.basename(img_path)
    parts = filename.split("_")
    patient_id = "_".join(parts[:3])   # BraTS20_Training_001
    patient_to_files.setdefault(patient_id, []).append(filename)

patients = sorted(patient_to_files.keys())
random.shuffle(patients)

n = len(patients)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_patients = patients[:train_end]
val_patients = patients[train_end:val_end]
test_patients = patients[val_end:]

def collect_files(patient_list):
    files = []
    for patient_id in patient_list:
        files.extend(patient_to_files[patient_id])
    return sorted(files)

train_files = collect_files(train_patients)
val_files = collect_files(val_patients)
test_files = collect_files(test_patients)

def save_list(file_list, out_path):
    with open(out_path, "w") as f:
        for item in file_list:
            f.write(item + "\n")

save_list(train_files, os.path.join(SPLIT_DIR, "train.txt"))
save_list(val_files, os.path.join(SPLIT_DIR, "val.txt"))
save_list(test_files, os.path.join(SPLIT_DIR, "test.txt"))

print("Total patients:", len(patients))
print("Train patients:", len(train_patients))
print("Val patients:", len(val_patients))
print("Test patients:", len(test_patients))

print("Train slices:", len(train_files))
print("Val slices:", len(val_files))
print("Test slices:", len(test_files))

print("\nSaved files:")
print(os.path.join(SPLIT_DIR, "train.txt"))
print(os.path.join(SPLIT_DIR, "val.txt"))
print(os.path.join(SPLIT_DIR, "test.txt"))