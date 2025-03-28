import torch
from torch.utils.data import DataLoader, Subset
from dataset import BioristorDataset
from model import BioristorModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model_path, test_indices_path, data_csv, data_image_dir, batch_size=32):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the model
    model = BioristorModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create full dataset
    full_dataset = BioristorDataset(
        csv_file=data_csv,
        root_dir=data_image_dir
    )
    
    # Load test indices
    test_indices = torch.load(test_indices_path)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Testing on {len(test_dataset)} samples")
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Lists to store predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Test the model
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
    plot_confusion_matrix(cm, class_names)
    
    # Calculate accuracy per class
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == all_labels[class_mask])
            print(f"{class_name} accuracy: {class_acc:.4f}")

def main():
    # Paths to your data and model
    model_path = 'best_model.pth'
    test_indices_path = 'test_indices.pt'
    data_csv = 'data/mapped_data.csv'
    data_image_dir = 'data/images'
    
    test_model(model_path, test_indices_path, data_csv, data_image_dir)

if __name__ == '__main__':
    main() 