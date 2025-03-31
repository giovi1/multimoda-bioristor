import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for the multimodal ResNet-LSTM model."""
    num_classes: int = 4              # Number of output classes
    image_size: int = 224             # Input image size
    num_images: int = 6               # Number of images per sample
    hidden_size: int = 128            # LSTM hidden size
    num_layers: int = 2               # Number of LSTM layers
    bidirectional: bool = True        # Whether to use bidirectional LSTM
    dropout: float = 0.2              # Dropout rate
    resnet_type: str = "resnet18"     # Type of ResNet (resnet18, resnet34, resnet50)
    freeze_resnet: bool = False       # Whether to freeze ResNet weights

class BasicBlock(nn.Module):
    """Basic ResNet block with skip connections."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection and apply ReLU
        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    """Custom ResNet implementation for image processing."""
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # Create downsample layer if stride != 1 or in_channels != out_channels
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block may have different stride and downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

def create_resnet(resnet_type):
    """Create ResNet model based on specified type."""
    if resnet_type == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2])
    elif resnet_type == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3])
    else:
        raise ValueError(f"Unsupported ResNet type: {resnet_type}")

class MultimodalResNetLSTM(nn.Module):
    """
    Multimodal architecture combining ResNet for images and LSTM for sensor data.
    
    The model processes:
    1. Images through a ResNet to extract visual features
    2. Sensor data through an LSTM to capture temporal patterns
    3. Combines both features for final classification
    """
    def __init__(self, config):
        super(MultimodalResNetLSTM, self).__init__()
        self.config = config
        
        # Image processing branch using ResNet
        self.resnet = create_resnet(config.resnet_type)
        
        # If using pretrained ResNet, replace here
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        
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