"""
Data Module for Chest X-ray Classification
This file loads and preprocesses the chest X-ray data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ChestXrayDataset(Dataset):
    """Dataset class for chest X-ray images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load all images and labels
        for class_name in ['NORMAL', 'PNEUMONIA']:
            label = 0 if class_name == 'NORMAL' else 1
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.jpeg'):
                self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} images from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_root='chest_xray', batch_size=32, image_size=224):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_root: Root directory with train/val_new/test folders
        batch_size: Batch size for loading
        image_size: Size to resize images (default 224x224)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Training transforms (WITH augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Val/Test transforms (NO augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ChestXrayDataset(f'{data_root}/train', train_transform)
    val_dataset = ChestXrayDataset(f'{data_root}/val_new', test_transform)  # FIXED: changed from 'val' to 'val_new'
    test_dataset = ChestXrayDataset(f'{data_root}/test', test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"\n✅ Data loaders created!")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    print(f"   Test:  {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader


# Test the module
if __name__ == "__main__":
    print("Testing data module...\n")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Load one batch
    images, labels = next(iter(train_loader))
    
    print(f"\n✅ Successfully loaded a batch!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"\n✅ Data module is working correctly!")