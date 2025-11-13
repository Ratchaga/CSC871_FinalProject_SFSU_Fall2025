import os
from pathlib import Path

# Set path to your dataset
data_dir = Path("chest_xray")

splits = ['train', 'test', 'val']
classes = ['NORMAL', 'PNEUMONIA']

print("Dataset Statistics:")
print("=" * 50)

for split in splits:
    print(f"\n{split.upper()} SET:")
    total = 0
    for cls in classes:
        path = data_dir / split / cls
        count = len(list(path.glob('*.jpeg')))
        print(f"  {cls}: {count} images")
        total += count
    print(f"  TOTAL: {total} images")