import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    num_classes: int = 4
    image_size: int = 224
    num_images: int = 6
    sensor_hidden_dims: list = (128, 64)
    dropout: float = 0.2

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class MultimodalResNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultimodalResNet, self).__init__()
        self.config = config
        
        # Image processing branch (ResNet18-like architecture)
        self.image_encoder = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256)
        
        # Sensor processing branch
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, config.sensor_hidden_dims[0]),
            nn.BatchNorm1d(config.sensor_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.sensor_hidden_dims[0], config.sensor_hidden_dims[1]),
            nn.BatchNorm1d(config.sensor_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + config.sensor_hidden_dims[1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )
        
    def forward(self, images, sensor_data):
        batch_size = images.size(0)
        
        # Process each image through ResNet
        image_features = []
        for i in range(self.config.num_images):
            img = images[:, i]  # (batch_size, 3, image_size, image_size)
            feat = self.image_encoder(img)  # (batch_size, 256)
            image_features.append(feat)
        
        # Average image features
        image_features = torch.stack(image_features, dim=1)  # (batch_size, num_images, 256)
        image_features = torch.mean(image_features, dim=1)  # (batch_size, 256)
        
        # Process sensor data
        sensor_features = self.sensor_encoder(sensor_data)  # (batch_size, sensor_hidden_dims[1])
        
        # Combine features
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output 