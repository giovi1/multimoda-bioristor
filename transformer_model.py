import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    num_classes: int = 4
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    image_size: int = 224
    num_images: int = 6

class CustomCNN(nn.Module):
    def __init__(self, output_dim):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to desired output dimension
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # Apply convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final projection
        x = self.fc(x)
        return x

class MultiModalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiModalTransformer, self).__init__()
        self.config = config
        
        # Sensor embedding
        self.sensor_embedding = nn.Sequential(
            nn.Linear(4, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Image embedding using custom CNN
        self.image_embedding = CustomCNN(config.d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, config.num_images + 1, config.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.LayerNorm(config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.num_classes)
        )
        
    def forward(self, images, sensor_data):
        batch_size = images.size(0)
        
        # Process sensor data
        sensor_features = self.sensor_embedding(sensor_data)  # (batch_size, d_model)
        
        # Process each image
        image_features = []
        for i in range(self.config.num_images):
            img = images[:, i]  # (batch_size, 3, image_size, image_size)
            feat = self.image_embedding(img)  # (batch_size, d_model)
            image_features.append(feat)
        
        # Combine all features
        combined_features = torch.stack([sensor_features] + image_features, dim=1)  # (batch_size, num_images+1, d_model)
        
        # Add positional encoding
        combined_features = combined_features + self.pos_encoder
        
        # Transformer processing
        transformer_output = self.transformer_encoder(combined_features)  # (batch_size, num_images+1, d_model)
        
        # Global average pooling
        pooled_output = torch.mean(transformer_output, dim=1)  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(pooled_output)
        
        return output 