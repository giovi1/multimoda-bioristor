import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, save_path):
    """
    Plot confusion matrix as a heatmap and save it.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        classes (list): List of class names
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_features(model, test_loader, device, output_dir, num_classes):
    """
    Extract and visualize features from both branches using t-SNE.
    
    Args:
        model: Model to extract features from
        test_loader: DataLoader for test data
        device: Device to run model on
        output_dir: Directory to save visualizations
        num_classes: Number of classes
    """
    try:
        # Create new model with feature extraction hooks
        class FeatureExtractor(torch.nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
                self.image_features = []
                self.sensor_features = []
                
            def forward(self, images, sensor_data):
                batch_size = images.size(0)
                
                # Process images with ResNet
                num_images = images.size(1)
                images = images.reshape(-1, 3, self.model.config.image_size, self.model.config.image_size)
                image_features = self.model.resnet(images)
                
                # Reshape back and average across images
                image_features = image_features.view(batch_size, num_images, -1)
                image_features = torch.mean(image_features, dim=1)
                
                # Store image features
                self.image_features.append(image_features.detach().cpu())
                
                # Process sensor data with LSTM
                sensor_data = sensor_data.unsqueeze(1)
                lstm_out, _ = self.model.lstm(sensor_data)
                sensor_features = lstm_out[:, -1, :]
                
                # Store sensor features
                self.sensor_features.append(sensor_features.detach().cpu())
                
                # Combine features and classify
                combined_features = torch.cat([image_features, sensor_features], dim=1)
                output = self.model.classifier(combined_features)
                
                return output
        
        # Create feature extractor model
        feature_extractor = FeatureExtractor(model)
        feature_extractor.eval()
        
        # Collect features and labels
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                sensor_data = batch['sensor_data'].to(device)
                images = batch['images'].to(device)
                labels = batch['label']
                
                _ = feature_extractor(images, sensor_data)
                all_labels.extend(labels.numpy())
        
        # Concatenate all features
        image_features = torch.cat(feature_extractor.image_features, dim=0).numpy()
        sensor_features = torch.cat(feature_extractor.sensor_features, dim=0).numpy()
        combined_features = np.concatenate([image_features, sensor_features], axis=1)
        labels = np.array(all_labels)
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        
        # Plot image features
        image_tsne = tsne.fit_transform(image_features)
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            plt.scatter(image_tsne[labels == i, 0], image_tsne[labels == i, 1], label=f'Class {i}')
        plt.title('t-SNE of Image Features')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'image_features_tsne.png'))
        plt.close()
        
        # Plot sensor features
        sensor_tsne = tsne.fit_transform(sensor_features)
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            plt.scatter(sensor_tsne[labels == i, 0], sensor_tsne[labels == i, 1], label=f'Class {i}')
        plt.title('t-SNE of Sensor Features')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'sensor_features_tsne.png'))
        plt.close()
        
        # Plot combined features
        combined_tsne = tsne.fit_transform(combined_features)
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            plt.scatter(combined_tsne[labels == i, 0], combined_tsne[labels == i, 1], label=f'Class {i}')
        plt.title('t-SNE of Combined Features')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'combined_features_tsne.png'))
        plt.close()
        
        print("Feature visualization completed and saved.")
        
    except ImportError:
        print("Skipping feature visualization - sklearn not available.") 