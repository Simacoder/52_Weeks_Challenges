# YOLOv1 Paper Walkthrough: The Day YOLO First Saw the World

A detailed walkthrough of the YOLOv1 architecture and its PyTorch implementation from scratch.

## Introduction

Object detection has been revolutionized by numerous breakthrough models, but one name stands out: YOLO. The very first version, YOLOv1, was introduced in 2015 through the seminal paper "You Only Look Once: Unified, Real-Time Object Detection" [1].

Before YOLOv1, the state-of-the-art approach was R-CNN (Region-based Convolutional Neural Network), which employed a multi-stage pipeline. This method used selective search to generate region proposals, extracted features via CNN, and classified objects using SVM. This process was computationally expensive and slow.

YOLOv1 changed the game by introducing a unified, single-stage detection framework that prioritized speed without sacrificing accuracy. While YOLOv13 has since been released, this project focuses on understanding the original architecture—where it all began. This repository provides a comprehensive walkthrough of how YOLOv1 works and includes a complete PyTorch implementation built from scratch.

## Table of Contents

- [Key Concepts](#key-concepts)
- [Architecture Overview](#architecture-overview)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [References](#references)

## Key Concepts

### The Grid-Based Approach

YOLOv1 divides an input image into an S×S grid (default: 7×7, producing 49 cells). Each grid cell is responsible for detecting objects whose center falls within that cell. This spatial division transforms object detection into a regression problem.

### Target Vector

For each grid cell, a target vector contains 25 elements:
- **Indices 0-19**: Class labels in one-hot encoding (for PASCAL VOC's 20 classes)
- **Index 20**: Confidence score (1 if object present, 0 otherwise)
- **Indices 21-22**: Normalized (x, y) coordinates of the object's midpoint
- **Indices 23-24**: Normalized width (w) and height (h) of the bounding box

### Prediction Vector

The prediction vector has 30 elements per cell because YOLOv1 predicts two bounding boxes per cell:
- **Classes**: 20 elements (same as target)
- **First bounding box**: 5 elements (x, y, w, h, confidence)
- **Second bounding box**: 5 elements (x, y, w, h, confidence)

During inference, only the bounding box with higher confidence is retained.

## Architecture Overview

### Network Design

YOLOv1 employs a CNN-based backbone followed by fully-connected layers. The complete architecture consists of:

- **24 Convolutional Layers**: Form the backbone, extracting spatial features from the input image
- **2 Fully-Connected Layers**: Perform final feature aggregation and prediction
- **Output Shape**: 30×7×7 tensor, where each of the 49 grid cells contains a 30-element prediction vector

### Building Blocks

**ConvBlock**: The fundamental building unit containing:
- A convolutional layer
- Leaky ReLU activation function
- Optional max-pooling layer

**Backbone**: A sequence of ConvBlocks that progressively extracts features from the 448×448 input image.

**Fully-Connected Layers**: Two linear layers with dropout (rate: 0.5) between them for regularization. Leaky ReLU is applied after the first layer, while the second layer (output layer) has no activation function.

### Assembly

The complete YOLOv1 model integrates the backbone and fully-connected components. The backbone output is flattened to create a 1D vector compatible with the fully-connected input, ultimately producing the 30×7×7 prediction tensor.

## Implementation Details

### Input Specifications

- **Batch Size**: Variable (typically 1 or more)
- **Channels**: 3 (RGB)
- **Dimensions**: 448×448 pixels

### Forward Pass

```python
yolov1 = YOLOv1()
x = torch.randn(1, 3, 448, 448)  # Random input tensor
out = yolov1(x)                   # Forward pass
out = out.reshape(-1, C+B*5, S, S)  # Reshape to grid format
```

Where:
- C = number of classes (20 for PASCAL VOC)
- B = number of bounding boxes per cell (2 for YOLOv1)
- S = grid size (7 for YOLOv1)

### Output Shape

The final output tensor has shape `(batch_size, 30, 7, 7)`, where:
- 30 = 20 classes + (2 boxes × 5 parameters)
- 7×7 = spatial grid

## Project Structure

```
yolov1-implementation/
├── README.md
├── model.py              # YOLOv1 model definition
├── blocks.py             # ConvBlock and component definitions
├── backbone.py           # Backbone architecture
├── fc_layers.py          # Fully-connected layers
├── train.py              # Training script
├── inference.py          # Inference and visualization
└── data/                 # Dataset directory
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib (for visualization)

### Installation

```bash
git clone <repository-url>
cd yolov1-implementation
pip install -r requirements.txt
```

### Quick Start

```python
import torch
from model import YOLOv1

# Initialize model
model = YOLOv1()

# Create sample input
x = torch.randn(1, 3, 448, 448)

# Forward pass
predictions = model(x)
print(predictions.shape)  # Output: torch.Size([1, 30, 7, 7])
```

### Training

```bash
python train.py --epochs 135 --batch-size 64 --learning-rate 1e-2
```

### Inference

```bash
python inference.py --image-path path/to/image.jpg --model-weights yolov1.pth
```

## Model Performance

YOLOv1 achieves a balance between speed and accuracy:
- **Speed**: Processes images in real-time (45 FPS)
- **Accuracy**: Competitive mAP on PASCAL VOC dataset
- **Advantages**: Unified, end-to-end training; global image context
- **Limitations**: Struggles with small objects; only two bounding boxes per cell

## References

[1] Joseph Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection." arXiv preprint arXiv:1506.02640 (2015). https://arxiv.org/pdf/1506.02640

[2] Ross Girshick et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." IEEE International Conference on Computer Vision (ICCV). 2014. https://arxiv.org/pdf/1311.2524

[3] Mengqi Lei et al. "YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception." arXiv preprint arXiv:2506.17733 (2025). https://arxiv.org/abs/2506.17733

[4] Bing Xu et al. "Empirical Evaluation of Rectified Activations in Convolutional Networks." arXiv preprint arXiv:1505.00853 (2015). https://arxiv.org/pdf/1505.00853

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the implementation, documentation, or add new features.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

This implementation follows the original YOLOv1 paper and is created for educational purposes to understand the foundational architecture of modern object detection systems.

## Author
- Simanga Mchunu