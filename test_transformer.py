import torch
from torch.utils.data import DataLoader, Subset
from dataset import BioristorDataset
from transformer_model import MultiModalTransformer, ModelConfig
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multimodal Transformer Model')
    
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
    parser.add_argument('--d_model', type=int, default=256,
                      help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                      help='Feed-forward network dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--num_images', type=int, default=6,
                      help='Number of images per sample')
    
    # Testing arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Directory containing the model checkpoint')
    parser.add_argument('--device', type=str, default='mps',
                      help='Device to use (cuda, mps, or cpu)')
    
    # Output arguments
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

def test_model(model, test_loader, device, output_dir):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            sensor_data = batch['sensor_data'].to(device)
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, sensor_data)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Print classification report
    class_names = ['healthy', 'uncertain', 'stress', 'recovery']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Calculate accuracy per class
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == all_labels[class_mask])
            print(f"{class_name} accuracy: {class_acc:.4f}")

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
        args.experiment_name = os.path.basename(args.checkpoint_dir)
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
    
    print(f"Testing on {len(test_dataset)} samples")
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Create model config
    config = ModelConfig(
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        image_size=args.image_size,
        num_images=args.num_images
    )
    
    # Create and load model
    model = MultiModalTransformer(config).to(device)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test the model
    test_model(model, test_loader, device, output_dir)

if __name__ == '__main__':
    main() 