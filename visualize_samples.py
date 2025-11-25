import torch
import torchvision
import matplotlib.pyplot as plt
import sys

# Add your project directory to sys.path so Python can find data_module
sys.path.append(r"C:\Users\91638\Desktop\CSC871_FinalProject_SFSU_Fall2025-main\CSC871_FinalProject_SFSU_Fall2025-main")

from data_module import get_data_loaders

# Load training loader
train_loader, _, _ = get_data_loaders(
    data_root=r'chest_xray\chest_xray',
    batch_size=8,
    image_size=224
)

# Get a batch
images, labels = next(iter(train_loader))

# Make a grid
grid = torchvision.utils.make_grid(images, nrow=4)
grid = grid.permute(1, 2, 0)  # [C, H, W] → [H, W, C]

# Unnormalize
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
grid = grid * std + mean
grid = grid.clip(0, 1)

# Plot and save
plt.figure(figsize=(10, 6))
plt.imshow(grid)
plt.axis('off')
plt.title("Sample Chest X-ray Images from Training Set")
plt.tight_layout()
plt.savefig("sample_images_train.png")
plt.show()
