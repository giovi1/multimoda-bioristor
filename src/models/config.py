from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Base configuration for all models."""
    num_classes: int = 4              # Number of output classes
    image_size: int = 224             # Input image size
    num_images: int = 6               # Number of images per sample
    dropout: float = 0.2              # Dropout rate

@dataclass
class ResNetLSTMConfig(ModelConfig):
    """Configuration for the ResNet-LSTM multimodal model."""
    hidden_size: int = 128            # LSTM hidden size
    num_layers: int = 2               # Number of LSTM layers
    bidirectional: bool = True        # Whether to use bidirectional LSTM
    resnet_type: str = "resnet18"     # Type of ResNet (resnet18, resnet34)
    freeze_resnet: bool = False       # Whether to freeze ResNet weights 