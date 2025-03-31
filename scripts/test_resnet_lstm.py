#!/usr/bin/env python
"""
Testing script for the ResNet-LSTM multimodal model.
"""
import argparse
import os
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.data import BioristorDataset, create_data_loaders, load_test_indices
from src.models import ResNetLSTMConfig, MultimodalResNetLSTM
from src.utils import get_device, plot_confusion_matrix, visualize_features

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Multimodal ResNet-LSTM Model')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, default='data/mapped_data.csv',
                      help='Path to the CSV file with sensor data')
    parser.add_argument('--image_dir', type=str, default='data/images',
                      help='Directory containing the images')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=4,
                      help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--num_images', type=int, default=6,
                      help='Number of images per sample')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate')
    parser.add_argument('--resnet_type', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34'],
                      help='Type of ResNet to use')
    
    # Testing arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Directory containing the model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for this experiment')
    parser.add_argument('--visualize_features', action='store_true',
                      help='Visualize image and sensor features using t-SNE')
    parser.add_argument('--class_names', type=str, nargs='+', 
                      default=['healthy', 'uncertain', 'stress', 'unknown'],
                      help='Names of classes')
    
    return parser.parse_args()

def test_model(model, test_loader, device):
    """
    Test the model on the test set.
    
    Args:
        model: Model to test
        test_loader: DataLoader for test data
        device: Device to test on
        
    Returns:
        all_preds: Predicted classes
        all_labels: True labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sensor_data = batch['sensor_data'].to(device)
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, sensor_data)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = get_device(args.device)
    print(f'Using device: {device}')
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = BioristorDataset(
        csv_file=args.data_csv,
        root_dir=args.image_dir
    )
    
    # Load test indices
    test_indices = load_test_indices(args.checkpoint_dir)
    if test_indices is None:
        print("Test indices not found, creating new random split.")
    
    # Create data loaders
    _, _, test_loader, _ = create_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_indices=test_indices
    )
    
    # Create model config
    config = ResNetLSTMConfig(
        num_classes=args.num_classes,
        image_size=args.image_size,
        num_images=args.num_images,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        resnet_type=args.resnet_type
    )
    
    # Create model and load checkpoint
    model = MultimodalResNetLSTM(config).to(device)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Visualize features if requested
    if args.visualize_features:
        visualize_features(model, test_loader, device, output_dir, args.num_classes)
    
    # Test the model
    all_preds, all_labels = test_model(model, test_loader, device)
    
    # Calculate accuracy
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Generate classification report
    class_names = args.class_names[:args.num_classes]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Calculate and save per-class accuracy
    per_class_acc = []
    for i in range(args.num_classes):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_preds)[class_mask] == np.array(all_labels)[class_mask])
            per_class_acc.append((class_names[i], class_acc * 100))
    
    print("\nPer-class Accuracy:")
    for class_name, acc in per_class_acc:
        print(f"{class_name}: {acc:.2f}%")
    
    # Save per-class accuracy
    with open(os.path.join(output_dir, 'per_class_accuracy.txt'), 'w') as f:
        f.write("Per-class Accuracy:\n")
        for class_name, acc in per_class_acc:
            f.write(f"{class_name}: {acc:.2f}%\n")
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == '__main__':
    main() 