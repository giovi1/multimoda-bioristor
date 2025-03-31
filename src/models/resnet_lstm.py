import torch
import torch.nn as nn
from src.models.resnet import create_resnet
from src.models.config import ResNetLSTMConfig

class MultimodalResNetLSTM(nn.Module):
    """
    Multimodal architecture combining ResNet for images and LSTM for sensor data.
    
    The model processes:
    1. Images through a ResNet to extract visual features
    2. Sensor data through an LSTM to capture temporal patterns
    3. Combines both features for final classification
    """
    def __init__(self, config: ResNetLSTMConfig):
        super(MultimodalResNetLSTM, self).__init__()
        self.config = config
        
        # Image processing branch using ResNet
        self.resnet = create_resnet(config.resnet_type)
        
        # Get ResNet feature dimension (512 for resnet18/34)
        self.resnet_out_dim = 512
        
        # Freeze ResNet weights if specified
        if config.freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Sensor processing branch using LSTM
        # Input shape: [batch_size, sequence_length, input_size]
        # For sensor data: sequence_length=1, input_size=4 (Rds, DIgs, tds, tgs)
        self.lstm = nn.LSTM(
            input_size=4,  # 4 sensor features
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # Calculate LSTM output dimension
        lstm_out_dim = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        
        # Feature fusion
        # Combine ResNet features and LSTM features
        combined_dim = self.resnet_out_dim + lstm_out_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes)
        )

    def forward(self, images, sensor_data):
        """
        Forward pass through the multimodal network.
        
        Args:
            images: Tensor of shape [batch_size, num_images, channels, height, width]
            sensor_data: Tensor of shape [batch_size, num_features]
            
        Returns:
            class_outputs: Tensor of shape [batch_size, num_classes]
        """
        batch_size = images.size(0)
        
        # Process images with ResNet
        # First, reshape to process all images through ResNet
        num_images = images.size(1)
        images = images.reshape(-1, 3, self.config.image_size, self.config.image_size)
        image_features = self.resnet(images)
        
        # Reshape back and average across images
        image_features = image_features.view(batch_size, num_images, -1)
        image_features = torch.mean(image_features, dim=1)  # Average across images
        
        # Process sensor data with LSTM
        # Add sequence dimension: [batch_size, 1, num_features]
        sensor_data = sensor_data.unsqueeze(1)
        lstm_out, _ = self.lstm(sensor_data)
        
        # Get last time step output
        # For bidirectional, concatenate forward and backward final states
        sensor_features = lstm_out[:, -1, :]
        
        # Combine features from both branches
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        
        # Final classification
        class_outputs = self.classifier(combined_features)
        
        return class_outputs 