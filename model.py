import torch
import torch.nn as nn
import torchvision.models as models

class BioristorModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BioristorModel, self).__init__()
        
        # Image processing branch (using ResNet50 as backbone)
        self.image_encoder = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.image_encoder.fc = nn.Identity()
        
        # Sensor data processing branch with more layers
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        # Combined processing with more layers
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 32, 1024),  # 2048 from ResNet50 + 32 from sensor
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, sensor_data):
        # Process images
        # Reshape images from (batch_size, 6, 3, 224, 224) to (batch_size*6, 3, 224, 224)
        batch_size = images.size(0)
        images = images.view(-1, 3, 224, 224)
        
        # Process each image through ResNet
        image_features = self.image_encoder(images)
        
        # Reshape back to (batch_size, 6, 2048)
        image_features = image_features.view(batch_size, 6, -1)
        
        # Average pooling across the 6 images
        image_features = torch.mean(image_features, dim=1)
        
        # Process sensor data
        sensor_features = self.sensor_encoder(sensor_data)
        
        # Combine features
        combined_features = torch.cat([image_features, sensor_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        return output 