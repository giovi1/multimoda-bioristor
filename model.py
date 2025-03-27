import torch
import torch.nn as nn
import torchvision.models as models

class BioristorModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BioristorModel, self).__init__()
        
        # Image processing branch (using ResNet18 as backbone)
        self.image_encoder = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.image_encoder.fc = nn.Identity()
        
        # Sensor data processing branch
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined processing
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 256),  # 512 from ResNet + 32 from sensor
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images, sensor_data):
        # Process images
        # Reshape images from (batch_size, 6, 3, 224, 224) to (batch_size*6, 3, 224, 224)
        batch_size = images.size(0)
        images = images.view(-1, 3, 224, 224)
        
        # Process each image through ResNet
        image_features = self.image_encoder(images)
        
        # Reshape back to (batch_size, 6, 512)
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