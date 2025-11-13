import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path

data_dir = Path("chest_xray/train")

def show_samples(class_name, num_samples=5):
    image_dir = data_dir / class_name
    image_files = list(image_dir.glob('*.jpeg'))
    samples = random.sample(image_files, num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle(f'{class_name} X-ray Samples', fontsize=16)
    
    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        # Show image size
        ax.set_title(f'{img.size[0]}x{img.size[1]}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{class_name}_samples.png')
    plt.show()
    print(f"✅ Saved {class_name}_samples.png")

# Show samples from each class
show_samples('NORMAL', 5)
show_samples('PNEUMONIA', 5)