import torch
from unet import UNet

model = UNet(in_channels=1, out_channels=1)
x = torch.randn(2, 1, 224, 224)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)