import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def setup_directory_structure():
    """Create the required directory structure for YOLOv1 dataset."""
    
    base_dirs = [
        'data/images/train',
        'data/images/val',
        'data/annotations/train',
        'data/annotations/val',
        'checkpoints',
        'test_images',
        'detections'
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    print("\n" + "="*60)
    print("Directory structure created successfully!")
    print("="*60)


def create_dummy_dataset(num_train=10, num_val=2):
    """
    Create a dummy dataset with sample images and annotations for testing.
    
    Args:
        num_train: Number of training images
        num_val: Number of validation images
    """
    
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    print("\nCreating dummy dataset for testing...")
    print("="*60)
    
    # Create training images and annotations
    for i in range(num_train):
        img_path = f'data/images/train/image_{i:03d}.jpg'
        ann_path = f'data/annotations/train/image_{i:03d}.xml'
        
        # Create random image
        image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
        cv2.imwrite(img_path, image)
        
        # Create random annotations
        height, width = 640, 480
        num_objects = np.random.randint(1, 4)
        
        root = ET.Element('annotation')
        
        filename = ET.SubElement(root, 'filename')
        filename.text = f'image_{i:03d}.jpg'
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        # Add random objects
        for j in range(num_objects):
            obj = ET.SubElement(root, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = CLASSES[np.random.randint(0, len(CLASSES))]
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = np.random.randint(10, width - 100)
            ymin = np.random.randint(10, height - 100)
            xmax = xmin + np.random.randint(50, 150)
            ymax = ymin + np.random.randint(50, 150)
            
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        
        # Save XML
        tree = ET.ElementTree(root)
        tree.write(ann_path)
        
        print(f"✓ Created train image: {img_path}")
    
    # Create validation images and annotations
    for i in range(num_val):
        img_path = f'data/images/val/image_val_{i:03d}.jpg'
        ann_path = f'data/annotations/val/image_val_{i:03d}.xml'
        
        # Create random image
        image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
        cv2.imwrite(img_path, image)
        
        # Create random annotations
        height, width = 640, 480
        num_objects = np.random.randint(1, 4)
        
        root = ET.Element('annotation')
        
        filename = ET.SubElement(root, 'filename')
        filename.text = f'image_val_{i:03d}.jpg'
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        # Add random objects
        for j in range(num_objects):
            obj = ET.SubElement(root, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = CLASSES[np.random.randint(0, len(CLASSES))]
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = np.random.randint(10, width - 100)
            ymin = np.random.randint(10, height - 100)
            xmax = xmin + np.random.randint(50, 150)
            ymax = ymin + np.random.randint(50, 150)
            
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        
        # Save XML
        tree = ET.ElementTree(root)
        tree.write(ann_path)
        
        print(f"✓ Created val image: {img_path}")
    
    print("\n" + "="*60)
    print(f"Dummy dataset created: {num_train} train + {num_val} val images")
    print("="*60)


def verify_dataset():
    """Verify that dataset is properly set up."""
    
    print("\nVerifying dataset structure...")
    print("="*60)
    
    train_images = list(Path('data/images/train').glob('*.jpg'))
    train_annot = list(Path('data/annotations/train').glob('*.xml'))
    val_images = list(Path('data/images/val').glob('*.jpg'))
    val_annot = list(Path('data/annotations/val').glob('*.xml'))
    
    print(f"Training images: {len(train_images)}")
    print(f"Training annotations: {len(train_annot)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Validation annotations: {len(val_annot)}")
    
    if len(train_images) == 0:
        print("\n⚠️  WARNING: No training images found!")
        return False
    
    print("\n" + "="*60)
    print("✓ Dataset structure verified successfully!")
    print("="*60)
    return True


def test_dataset_loading():
    """Test if the dataset can be loaded correctly."""
    
    print("\nTesting dataset loading...")
    print("="*60)
    
    try:
        from dataset import YOLODataset, collate_fn
        from torch.utils.data import DataLoader
        
        # Load dataset
        dataset = YOLODataset(
            data_dir='./data',
            split='train',
            img_size=448,
            augment=False
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Dataset size: {len(dataset)}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn
        )
        print(f"✓ DataLoader created successfully!")
        
        # Test batch
        images, targets = next(iter(dataloader))
        print(f"✓ Batch loaded successfully!")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets shape: {targets.shape}")
        
        print("\n" + "="*60)
        print("✓ Dataset test passed! Ready for training!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_directory_tree():
    """Print the directory structure."""
    
    print("\nDirectory structure:")
    print("="*60)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        try:
            entries = sorted(Path(directory).iterdir())
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                current_prefix = "└── " if is_last else "├── "
                print(prefix + current_prefix + entry.name)
                
                if entry.is_dir() and entry.name not in ['.git', '__pycache__']:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(entry, next_prefix, max_depth, current_depth + 1)
        except PermissionError:
            pass
    
    print_tree(".")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLOv1 Dataset Setup Script")
    print("="*60)
    
    # Step 1: Create directory structure
    print("\nStep 1: Creating directory structure...")
    setup_directory_structure()
    
    # Step 2: Create dummy dataset
    print("\nStep 2: Creating dummy dataset for testing...")
    create_dummy_dataset(num_train=10, num_val=2)
    
    # Step 3: Verify dataset
    print("\nStep 3: Verifying dataset...")
    if verify_dataset():
        # Step 4: Test loading
        print("\nStep 4: Testing dataset loading...")
        if test_dataset_loading():
            # Step 5: Print structure
            print("\nStep 5: Directory structure:")
            print_directory_tree()
            
            print("\n" + "="*60)
            print("✓ Setup complete! You can now:")
            print("  1. Replace dummy data with real images/annotations")
            print("  2. Run: python train.py --epochs 10 --batch-size 4")
            print("  3. Run: python inference.py --model-path <path> --image-path <path>")
            print("="*60)