import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Assuming you have this module defined
# from model import YOLOv1

class YOLOv1Detector:
    """
    YOLOv1 object detector for inference.
    """
    def __init__(self, model_path, S=7, B=2, C=20, conf_threshold=0.5, nms_threshold=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.S = S
        self.B = B
        self.C = C
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load model
        try:
            from model import YOLOv1
        except ImportError:
            raise ImportError("Please ensure model.py is in the same directory")
        
        self.model = YOLOv1(S=S, B=B, C=C)
        self.model = self.model.to(self.device)
        self.load_checkpoint(model_path)
        self.model.eval()
        
        # COCO class names (adjust for your dataset)
        self.class_names = self._get_class_names()

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Model loaded from {checkpoint_path}')

    def _get_class_names(self):
        """
        PASCAL VOC classes. Adjust if using different dataset.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def preprocess(self, image_path, img_size=448):
        """
        Load and preprocess image.
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            image = image_path
        
        original_h, original_w = image.shape[:2]
        
        # Resize to model input size
        image_resized = cv2.resize(image, (img_size, img_size))
        
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, image_resized, (original_w, original_h)

    def decode_predictions(self, predictions):
        """
        Decode model predictions to bounding boxes and class predictions.
        """
        predictions = predictions.permute(0, 2, 3, 1).contiguous()
        batch_size = predictions.size(0)
        
        all_boxes = []
        
        for b in range(batch_size):
            pred = predictions[b]  # (S, S, 30)
            boxes = []
            
            for i in range(self.S):
                for j in range(self.S):
                    cell = pred[i, j]
                    
                    # Extract class probabilities
                    class_probs = cell[self.B*5:]  # (C,)
                    class_probs = torch.softmax(class_probs, dim=0)
                    
                    # Extract bounding boxes
                    for b_idx in range(self.B):
                        start_idx = b_idx * 5
                        x = cell[start_idx]
                        y = cell[start_idx + 1]
                        w = cell[start_idx + 2]
                        h = cell[start_idx + 3]
                        conf = torch.sigmoid(cell[start_idx + 4])
                        
                        # Convert grid coordinates to image coordinates
                        x = (j + torch.sigmoid(x)) / self.S
                        y = (i + torch.sigmoid(y)) / self.S
                        w = torch.exp(w) / self.S
                        h = torch.exp(h) / self.S
                        
                        # Get class with highest probability
                        class_idx = torch.argmax(class_probs)
                        class_prob = class_probs[class_idx]
                        
                        # Combined score
                        score = conf * class_prob
                        
                        if score > self.conf_threshold:
                            boxes.append({
                                'x': x.item(),
                                'y': y.item(),
                                'w': w.item(),
                                'h': h.item(),
                                'conf': conf.item(),
                                'class_prob': class_prob.item(),
                                'score': score.item(),
                                'class_idx': class_idx.item(),
                                'class_name': self.class_names[class_idx.item()]
                            })
            
            all_boxes.append(boxes)
        
        return all_boxes

    def nms(self, boxes, threshold=0.4):
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        """
        if len(boxes) == 0:
            return []
        
        # Sort by score
        boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
        keep = []
        
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            remaining = []
            for box in boxes:
                # Only apply NMS to same class
                if box['class_idx'] != current['class_idx']:
                    remaining.append(box)
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(current, box)
                
                if iou < threshold:
                    remaining.append(box)
            
            boxes = remaining
        
        return keep

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two boxes.
        """
        # Convert from center coordinates to corner coordinates
        x1_min = box1['x'] - box1['w'] / 2
        y1_min = box1['y'] - box1['h'] / 2
        x1_max = box1['x'] + box1['w'] / 2
        y1_max = box1['y'] + box1['h'] / 2
        
        x2_min = box2['x'] - box2['w'] / 2
        y2_min = box2['y'] - box2['h'] / 2
        x2_max = box2['x'] + box2['w'] / 2
        y2_max = box2['y'] + box2['h'] / 2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = box1['w'] * box1['h']
        box2_area = box2['w'] * box2['h']
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def detect(self, image_path):
        """
        Perform object detection on an image.
        """
        image_tensor, image_resized, original_size = self.preprocess(image_path)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        boxes = self.decode_predictions(predictions)
        boxes = self.nms(boxes[0], threshold=self.nms_threshold)
        
        return boxes, image_resized, original_size

    def visualize(self, boxes, image, output_path=None, conf_threshold=None):
        """
        Draw bounding boxes on image and display/save.
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # Display image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        
        h, w = image.shape[:2]
        
        # Draw boxes
        for box in boxes:
            if box['score'] < conf_threshold:
                continue
            
            x_center = box['x'] * w
            y_center = box['y'] * h
            box_w = box['w'] * w
            box_h = box['h'] * h
            
            x_min = x_center - box_w / 2
            y_min = y_center - box_h / 2
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), box_w, box_h,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{box['class_name']}: {box['score']:.2f}"
            ax.text(x_min, y_min - 5, label, color='red', fontsize=10,
                   bbox=dict(facecolor='yellow', alpha=0.7))
        
        ax.set_title(f'YOLOv1 Detection (Detections: {len(boxes)})')
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f'Result saved to {output_path}')
        
        plt.tight_layout()
        return fig


def main():
    parser = argparse.ArgumentParser(description='YOLOv1 Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-path', type=str, default='./output.jpg', help='Path to save output image')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--display', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    # Initialize detector
    print('Loading YOLOv1 detector...')
    detector = YOLOv1Detector(
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # Perform detection
    print(f'Running inference on {args.image_path}...')
    boxes, image, original_size = detector.detect(args.image_path)
    
    print(f'Found {len(boxes)} objects:')
    for i, box in enumerate(boxes, 1):
        print(f"  {i}. {box['class_name']}: {box['score']:.4f}")
    
    # Visualize results
    detector.visualize(boxes, image, output_path=args.output_path)
    
    if args.display:
        plt.show()


if __name__ == '__main__':
    main()