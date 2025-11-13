import shutil
from pathlib import Path
import random

# Set seed so results are reproducible
random.seed(42)

data_dir = Path("chest_xray")
train_dir = data_dir / "train"
new_val_dir = data_dir / "val_new"

# Create new validation directory
new_val_dir.mkdir(exist_ok=True)

classes = ['NORMAL', 'PNEUMONIA']

for cls in classes:
    # Get all training images for this class
    train_cls_dir = train_dir / cls
    images = list(train_cls_dir.glob('*.jpeg'))
    
    # Shuffle them
    random.shuffle(images)
    
    # Take 15% for validation
    num_val = int(len(images) * 0.15)
    val_images = images[:num_val]
    
    # Create validation class directory
    val_cls_dir = new_val_dir / cls
    val_cls_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy images to validation
    for img in val_images:
        shutil.copy(img, val_cls_dir / img.name)
    
    print(f"{cls}: Moved {num_val} images to validation")

print(f"\n✅ New validation set created at: {new_val_dir}")
print("Now you have:")
print("  - train: ~4,430 images")
print("  - val_new: ~780 images")
print("  - test: 624 images")