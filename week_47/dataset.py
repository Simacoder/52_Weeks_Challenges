import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YOLODataset(Dataset):
    """
    YOLOv1 Dataset class for loading PASCAL VOC format data.
    
    Expected directory structure:
    data/
    ├── images/
    │   ├── train/
    │   │   └── *.jpg
    │   └── val/
    │       └── *.jpg
    └── annotations/
        ├── train/
        │   └── *.xml
        └── val/
            └── *.xml
    """
    
    # PASCAL VOC class names (adjust if using different dataset)
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, data_dir, split='train', img_size=448, S=7, B=2, C=20, augment=False):
        """
        Args:
            data_dir (str): Root directory of dataset
            split (str): 'train' or 'val'
            img_size (int): Image size (default: 448)
            S (int): Grid size (default: 7)
            B (int): Number of bounding boxes per cell (default: 2)
            C (int): Number of classes (default: 20)
            augment (bool): Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = C
        self.augment = augment
        
        # Paths
        self.img_dir = self.data_dir / 'images' / split
        self.ann_dir = self.data_dir / 'annotations' / split
        
        # Get image files
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')))
        
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        # Data augmentation
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=10, p=0.3),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Image tensor of shape (3, 448, 448)
            target (torch.Tensor): Target tensor of shape (S, S, B*5+C)
        """
        img_path = self.img_files[idx]
        ann_path = self.ann_dir / (img_path.stem + '.xml')
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse annotations
        bboxes, labels = self._parse_annotation(ann_path, image.shape)
        
        # Normalize bounding boxes to [0, 1]
        h, w = image.shape[:2]
        normalized_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min, x_max = x_min / w, x_max / w
            y_min, y_max = y_min / h, y_max / h
            normalized_bboxes.append([x_min, y_min, x_max, y_max])
        
        # Apply augmentation
        if self.augment or self.transform:
            augmented = self.transform(
                image=image,
                bboxes=normalized_bboxes,
                class_labels=labels
            )
            image = augmented['image']
            normalized_bboxes = augmented['bboxes']
            labels = augmented['class_labels']
        else:
            # Manual normalization if no augmentation
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Create target tensor
        target = self._create_target(normalized_bboxes, labels)
        
        return image, target
    
    def _parse_annotation(self, ann_path, img_shape):
        """
        Parse XML annotation file in PASCAL VOC format.
        
        Returns:
            bboxes (list): List of [x_min, y_min, x_max, y_max] in pixel coordinates
            labels (list): List of class indices
        """
        bboxes = []
        labels = []
        
        if not ann_path.exists():
            # Return empty annotations if file doesn't exist
            return bboxes, labels
        
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            # Get class name and convert to index
            class_name = obj.find('name').text
            if class_name not in self.CLASSES:
                continue
            class_idx = self.CLASSES.index(class_name)
            
            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            
            # Clip to image boundaries
            h, w = img_shape[:2]
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))
            
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_idx)
        
        return bboxes, labels
    
    def _create_target(self, bboxes, labels):
        """
        Create target tensor from bounding boxes and labels.
        
        Target shape: (S, S, B*5+C)
        Where each cell contains:
        - B bounding boxes with 5 values each (x, y, w, h, confidence)
        - C class probabilities
        
        Args:
            bboxes (list): Normalized bounding boxes [x_min, y_min, x_max, y_max] in [0, 1]
            labels (list): Class indices
        
        Returns:
            target (torch.Tensor): Target tensor of shape (S, S, B*5+C)
        """
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)
        
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox
            
            # Convert to center coordinates and dimensions
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            
            # Find which grid cell the center falls into
            cell_x = int(x_center * self.S)
            cell_y = int(y_center * self.S)
            
            # Ensure cell is within grid
            cell_x = min(cell_x, self.S - 1)
            cell_y = min(cell_y, self.S - 1)
            
            # If cell already has an object, skip (or use first one)
            if target[cell_y, cell_x, 4] > 0:
                continue
            
            # Normalize coordinates relative to cell
            x_cell = x_center * self.S - cell_x
            y_cell = y_center * self.S - cell_y
            
            # Set bounding box (using first box slot, index 0)
            target[cell_y, cell_x, 0] = x_cell       # x coordinate
            target[cell_y, cell_x, 1] = y_cell       # y coordinate
            target[cell_y, cell_x, 2] = w            # width
            target[cell_y, cell_x, 3] = h            # height
            target[cell_y, cell_x, 4] = 1            # confidence (object present)
            
            # Set class probabilities
            target[cell_y, cell_x, self.B * 5 + label] = 1
        
        return target
    
    @staticmethod
    def get_class_name(idx):
        """Get class name from index."""
        return YOLODataset.CLASSES[idx] if idx < len(YOLODataset.CLASSES) else 'Unknown'
    
    @staticmethod
    def get_class_index(name):
        """Get class index from name."""
        return YOLODataset.CLASSES.index(name) if name in YOLODataset.CLASSES else -1


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles variable-sized annotations by padding target tensors.
    
    Args:
        batch (list): List of (image, target) tuples from dataset
    
    Returns:
        images (torch.Tensor): Stacked image tensors
        targets (torch.Tensor): Stacked target tensors
    """
    images, targets = zip(*batch)
    
    # Stack images (they're already the same size)
    images = torch.stack(images, dim=0)
    
    # Stack targets (they're already the same size)
    targets = torch.stack(targets, dim=0)
    
    # Permute to (batch_size, B*5+C, S, S) format for loss function
    targets = targets.permute(0, 3, 1, 2)
    
    return images, targets


# Example usage and testing
if __name__ == "__main__":
    # Test dataset creation
    print("Creating YOLOv1 dataset...")
    
    # Create dummy dataset structure for testing
    data_dir = Path('./data')
    
    # Check if dataset exists
    if not (data_dir / 'images' / 'train').exists():
        print("Dataset not found. Please prepare dataset in ./data/")
        print("\nExpected structure:")
        print("data/")
        print("├── images/")
        print("│   ├── train/")
        print("│   │   └── *.jpg")
        print("│   └── val/")
        print("│       └── *.jpg")
        print("└── annotations/")
        print("    ├── train/")
        print("    │   └── *.xml")
        print("    └── val/")
        print("        └── *.xml")
    else:
        # Load dataset
        train_dataset = YOLODataset(
            data_dir='./data',
            split='train',
            img_size=448,
            augment=True
        )
        
        print(f"Dataset size: {len(train_dataset)}")
        
        # Test a sample
        image, target = train_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Target shape: {target.shape}")
        
        # Create dataloader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Test batch
        images, targets = next(iter(dataloader))
        print(f"\nBatch:")
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")