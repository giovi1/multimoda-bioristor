import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from dataset import BioristorDataset
from resnet_multimodal import MultimodalResNet, ModelConfig
from tqdm import tqdm
import argparse
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multimodal ResNet Model')
    
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
    parser.add_argument('--sensor_hidden_dims', type=int, nargs='+', default=[128, 64],
                      help='Hidden dimensions for sensor network')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate')
    
    # Testing arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Directory containing the model checkpoint')
    parser.add_argument('--device', type=str, default='mps',
                      help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for this experiment')
    
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            sensor_data = batch['sensor_data'].to(device)
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, sensor_data)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * correct / total
    return test_loss / len(test_loader), test_acc, all_preds, all_labels

def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full dataset
    full_dataset = BioristorDataset(
        csv_file=args.data_csv,
        root_dir=args.image_dir
    )
    
    # Load test indices
    test_indices = torch.load(os.path.join(args.checkpoint_dir, 'test_indices.pt'))
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    
    # Create model config
    config = ModelConfig(
        num_classes=args.num_classes,
        image_size=args.image_size,
        num_images=args.num_images,
        sensor_hidden_dims=args.sensor_hidden_dims,
        dropout=args.dropout
    )
    
    # Create model and load checkpoint
    model = MultimodalResNet(config).to(device)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Define loss function
    criterion = CrossEntropyLoss()
    
    # Test the model
    test_loss, test_acc, all_preds, all_labels = test_model(
        model, test_loader, criterion, device
    )
    
    # Print results
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    
    # Generate classification report
    class_names = ['healthy', 'uncertain', 'stress']  # Update based on your classes
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

if __name__ == '__main__':
    main() 