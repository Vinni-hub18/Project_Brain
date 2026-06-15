import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from unet import UNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SPLIT = os.path.join(BASE_DIR, "DATA", "splits", "train.txt")
VAL_SPLIT = os.path.join(BASE_DIR, "DATA", "splits", "val.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS_TO_RUN = 1
LR = 1e-4

CHECKPOINT_PATH = os.path.join(BASE_DIR, "model", "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")

train_dataset = BrainTumorDataset(BASE_DIR, TRAIN_SPLIT)
val_dataset = BrainTumorDataset(BASE_DIR, VAL_SPLIT)

train_dataset.file_names = train_dataset.file_names[:200]
val_dataset.file_names = val_dataset.file_names[:50]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
bce_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()

start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, start_epoch + EPOCHS_TO_RUN):
    model.train()
    train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE).float()
        masks = masks.to(DEVICE).float()

        optimizer.zero_grad()
        outputs = model(images)

        loss_bce = bce_loss(outputs, masks)
        loss_dice = dice_loss_from_logits(outputs, masks)
        loss = loss_bce + loss_dice

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                f"BCE: {loss_bce.item():.4f}, Dice: {loss_dice.item():.4f}, Total: {loss.item():.4f}"
            )

    train_loss /= max(len(train_loader), 1)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE).float()
            masks = masks.to(DEVICE).float()

            outputs = model(images)
            loss_bce = bce_loss(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice
            val_loss += loss.item()

    val_loss /= max(len(val_loader), 1)

    print(f"\nEpoch [{epoch + 1}] Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved to: {BEST_MODEL_PATH}")

print("Training complete.")