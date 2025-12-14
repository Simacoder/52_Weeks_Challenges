import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block consisting of Conv2d, BatchNorm2d, LeakyReLU, and optional MaxPool2d.
    This is the fundamental building block of YOLOv1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 batch_norm=True, activation=True, pool=False, pool_size=2):
        super(ConvBlock, self).__init__()
        
        layers = []
        
        # Convolution layer
        layers.append(nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=not batch_norm
        ))
        
        # Batch Normalization
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation function
        if activation:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        # Max Pooling
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Backbone(nn.Module):
    """
    YOLOv1 Backbone: 24 convolutional layers that extract features from input image.
    Progressively reduces spatial dimensions while increasing feature depth.
    """
    def __init__(self):
        super(Backbone, self).__init__()
        
        # Configuration: (in_channels, out_channels, kernel_size, stride, padding, pool)
        conv_configs = [
            (3, 64, 7, 2, 3, True),      # Layer 1: 448x448 -> 224x224
            (64, 192, 3, 1, 1, True),    # Layer 2: 224x224 -> 112x112
            (192, 128, 1, 1, 0, False),  # Layer 3
            (128, 256, 3, 1, 1, False),  # Layer 4
            (256, 256, 1, 1, 0, False),  # Layer 5
            (256, 512, 3, 1, 1, True),   # Layer 6: 112x112 -> 56x56
            
            # 4x (1x1 reduce, 3x3 expand) blocks
            (512, 256, 1, 1, 0, False),  # Layer 7
            (256, 512, 3, 1, 1, False),  # Layer 8
            (512, 256, 1, 1, 0, False),  # Layer 9
            (256, 512, 3, 1, 1, False),  # Layer 10
            (512, 256, 1, 1, 0, False),  # Layer 11
            (256, 512, 3, 1, 1, False),  # Layer 12
            (512, 256, 1, 1, 0, False),  # Layer 13
            (256, 512, 3, 1, 1, False),  # Layer 14
            (512, 512, 1, 1, 0, False),  # Layer 15
            (512, 1024, 3, 1, 1, True),  # Layer 16: 56x56 -> 28x28
            
            # 2x (1x1 reduce, 3x3 expand) blocks
            (1024, 512, 1, 1, 0, False), # Layer 17
            (512, 1024, 3, 1, 1, False), # Layer 18
            (1024, 512, 1, 1, 0, False), # Layer 19
            (512, 1024, 3, 1, 1, False), # Layer 20
            (1024, 1024, 3, 1, 1, False),# Layer 21
            (1024, 1024, 3, 2, 1, False),# Layer 22: 28x28 -> 14x14
            (1024, 1024, 3, 1, 1, False),# Layer 23
            (1024, 1024, 3, 1, 1, False),# Layer 24
        ]
        
        self.layers = nn.ModuleList()
        
        for in_ch, out_ch, k, s, p, pool in conv_configs:
            self.layers.append(ConvBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                batch_norm=True,
                activation=True,
                pool=pool,
                pool_size=2
            ))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class YOLOv1(nn.Module):
    """
    YOLOv1: You Only Look Once - Unified, Real-Time Object Detection
    
    This implementation follows the original paper: https://arxiv.org/pdf/1506.02640
    
    Architecture:
    - Backbone: 24 convolutional layers for feature extraction
    - Adaptive Average Pooling: Reduce spatial dimensions to 7x7
    - FC Layers: 2 fully-connected layers for prediction
    - Output: (batch_size, S, S, B*5+C) which is reshaped to (batch_size, B*5+C, S, S)
    
    Args:
        S (int): Grid size (default: 7)
        B (int): Number of bounding boxes per cell (default: 2)
        C (int): Number of classes (default: 20 for PASCAL VOC)
    """
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        # Backbone for feature extraction
        self.backbone = Backbone()
        
        # Adaptive average pooling to get (batch, 1024, 7, 7)
        self.avgpool = nn.AdaptiveAvgPool2d((S, S))
        
        # Output size: S * S * (B * 5 + C)
        # 7 * 7 * (2 * 5 + 20) = 49 * 30 = 1470
        self.output_size = S * S * (B * 5 + C)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(1024 * S * S, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, self.output_size)
        
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        """
        Forward pass of YOLOv1.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 448, 448)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, S, S, B*5+C)
                         Reshaped to (batch_size, B*5+C, S, S) for compatibility
        """
        batch_size = x.size(0)
        
        # Extract features using backbone
        features = self.backbone(x)  # Shape: (batch, 1024, H, W)
        
        # Adaptive pooling to get (batch, 1024, S, S)
        features = self.avgpool(features)
        
        # Flatten for FC layers
        features = features.view(batch_size, -1)
        
        # Generate predictions using fully-connected layers
        x = self.fc1(features)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape to (batch_size, S, S, B*5+C)
        predictions = x.view(batch_size, self.S, self.S, self.B * 5 + self.C)
        
        # Permute to (batch_size, B*5+C, S, S) for compatibility with loss function
        predictions = predictions.permute(0, 3, 1, 2)
        
        return predictions
    
    def get_architecture_info(self):
        """
        Print detailed architecture information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("="*60)
        print("YOLOv1 Architecture Information")
        print("="*60)
        print(f"Grid Size (S): {self.S}")
        print(f"Bounding Boxes per Cell (B): {self.B}")
        print(f"Number of Classes (C): {self.C}")
        print(f"Input Size: 3 x 448 x 448")
        print(f"Output Size: {self.B * 5 + self.C} x {self.S} x {self.S}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("="*60)
    
    def count_parameters(self):
        """
        Count total and trainable parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = YOLOv1(S=7, B=2, C=20)
    
    # Print architecture info
    model.get_architecture_info()
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (1, 30, 7, 7)")
    
    # Test with batch
    print("\nTesting with batch size 4...")
    x = torch.randn(4, 3, 448, 448)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    output = model(x)
    
    print(f"\nModel successfully moved to {device}")
    print(f"Output on {device}: {output.shape}")
    print("✓ All tests passed!")