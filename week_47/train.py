import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
from tqdm import tqdm
import json

# Assuming you have these modules defined
# from model import YOLOv1
# from dataset import YOLODataset, collate_fn

class YOLOLoss(nn.Module):
    """
    YOLOv1 Loss Function combining localization, confidence, and classification losses.
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord  # Weight for coordinate loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence loss
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 30, 7, 7) - model output
            targets: (batch_size, 30, 7, 7) - ground truth
        """
        batch_size = predictions.size(0)
        
        # Reshape to (batch_size, S, S, B*5 + C)
        pred = predictions.permute(0, 2, 3, 1).contiguous().view(batch_size, self.S, self.S, -1)
        target = targets.permute(0, 2, 3, 1).contiguous().view(batch_size, self.S, self.S, -1)

        # Extract components
        pred_boxes = pred[..., :self.B*5].view(batch_size, self.S, self.S, self.B, 5)
        pred_class = pred[..., self.B*5:]
        
        target_boxes = target[..., :self.B*5].view(batch_size, self.S, self.S, self.B, 5)
        target_class = target[..., self.B*5:]

        # Object mask (cells containing objects)
        obj_mask = target_boxes[..., 4] > 0  # confidence > 0
        noobj_mask = ~obj_mask

        loss = 0

        # 1. Bounding box regression loss (only for cells with objects)
        if obj_mask.sum() > 0:
            # Coordinate loss (x, y)
            loss += self.lambda_coord * self.mse(
                pred_boxes[obj_mask, :, :2],
                target_boxes[obj_mask, :, :2]
            )
            
            # Width and height loss (apply sqrt for stability)
            loss += self.lambda_coord * self.mse(
                torch.sqrt(torch.clamp(pred_boxes[obj_mask, :, 2:4], min=1e-6)),
                torch.sqrt(torch.clamp(target_boxes[obj_mask, :, 2:4], min=1e-6))
            )
            
            # Confidence loss for boxes with objects
            loss += self.mse(
                pred_boxes[obj_mask, :, 4],
                target_boxes[obj_mask, :, 4]
            )

        # 2. Confidence loss for boxes without objects
        if noobj_mask.sum() > 0:
            loss += self.lambda_noobj * self.mse(
                pred_boxes[noobj_mask, :, 4],
                torch.zeros_like(pred_boxes[noobj_mask, :, 4])
            )

        # 3. Classification loss
        if obj_mask.sum() > 0:
            loss += self.mse(
                pred_class[obj_mask[:, :, :, 0]],
                target_class[obj_mask[:, :, :, 0]]
            )

        return loss / batch_size


class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.criterion = YOLOLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
        
        # Create checkpoint directory
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.args.epochs}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        epoch_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(epoch_loss)
        
        return epoch_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating')
            
            for images, targets in progress_bar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                progress_bar.set_postfix({'val_loss': f'{total_loss / (len(progress_bar) + 1):.4f}'})
        
        epoch_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(epoch_loss)
        
        return epoch_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        checkpoint_path = Path(self.args.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.args.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'Best model saved: {best_path}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.history = checkpoint['history']
        print(f'Checkpoint loaded from {checkpoint_path}')

    def train(self):
        print(f'Training YOLOv1 on {self.device}')
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f'Epoch {epoch+1}/{self.args.epochs} - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            self.scheduler.step()
        
        # Save final model
        self.save_checkpoint(self.args.epochs - 1, val_loss)
        self._save_history()
        print('Training completed!')

    def _save_history(self):
        history_path = Path(self.args.checkpoint_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        print(f'Training history saved to {history_path}')


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv1 on custom dataset')
    parser.add_argument('--epochs', type=int, default=135, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Import model and dataset (adjust imports based on your project structure)
    try:
        from model import YOLOv1
        from dataset import YOLODataset, collate_fn
    except ImportError:
        print("Error: Please ensure model.py and dataset.py are in the same directory")
        return
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = YOLODataset(
        data_dir=args.data_dir,
        split='train',
        img_size=448,
        augment=True
    )
    val_dataset = YOLODataset(
        data_dir=args.data_dir,
        split='val',
        img_size=448,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print('Initializing YOLOv1 model...')
    model = YOLOv1(S=7, B=2, C=20)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, args)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()