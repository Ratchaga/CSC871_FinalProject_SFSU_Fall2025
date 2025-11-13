# Data Module - Ready for Training! 🎉

## Quick Start
```python
from data_module import get_data_loaders

# Get data loaders
train_loader, val_loader, test_loader = get_data_loaders(
    batch_size=32,
    image_size=224
)

# Use in your training loop:
for images, labels in train_loader:
    # images: shape [32, 3, 224, 224]
    # labels: 0 = NORMAL, 1 = PNEUMONIA
    # Your training code here
    pass
```

## Dataset Information

- **Training set**: 5,216 images
- **Validation set**: 782 images
- **Test set**: 624 images
- **Classes**: 
  - 0 = NORMAL
  - 1 = PNEUMONIA
- **Image size**: 224x224 pixels
- **Format**: RGB (3 channels)

## Data Augmentation

**Training data** includes:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness ±20%, contrast ±20%)

**Validation/Test data**: No augmentation (only resize + normalize)

## Class Imbalance

⚠️ **Important**: PNEUMONIA is ~3x more common than NORMAL

**Recommendation**: Use weighted loss function in training:
```python
# Example:
class_weights = torch.tensor([1.0, 0.33])  # Adjust as needed
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

## Files in This Project

- `data_module.py` - Main data loading code (use this!)
- `explore_data.py` - Script to count images
- `visualize_samples.py` - Visualize sample X-rays
- `create_val_split.py` - Script that created val_new folder
- `chest_xray/` - The dataset
  - `train/` - Training images
  - `val_new/` - Validation images (USE THIS, not val/)
  - `test/` - Test images

## Questions?

Contact: Fayeeza
