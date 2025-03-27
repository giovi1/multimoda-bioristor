import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class BioristorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with sensor data
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Default image transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Convert labels to numeric values (4 classes)
        self.label_map = {'healthy': 0, 'uncertain': 1, 'stress': 2, 'recovery': 3}
        
        # Calculate sensor statistics for normalization
        self.sensor_columns = ['Rds', 'DIgs', 'tds', 'tgs']
        self.sensor_means = {}
        self.sensor_stds = {}
        
        for col in self.sensor_columns:
            # Convert comma to dot and handle missing values
            values = self.data[col].apply(lambda x: float(x.replace(',', '.')) if pd.notna(x) else 0.0)
            self.sensor_means[col] = values.mean()
            self.sensor_stds[col] = values.std()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get and normalize sensor data
        sensor_data = []
        for col in self.sensor_columns:
            value = float(self.data.iloc[idx][col].replace(',', '.')) if pd.notna(self.data.iloc[idx][col]) else 0.0
            # Normalize using z-score normalization
            normalized_value = (value - self.sensor_means[col]) / self.sensor_stds[col]
            sensor_data.append(normalized_value)
            
        sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
        
        # Get label
        label = self.label_map[self.data.iloc[idx]['label']]
        
        # Get image paths
        image_paths = self.data.iloc[idx]['Image_Paths'].split(', ')
        
        # Load and transform images
        images = []
        for img_path in image_paths:
            full_path = os.path.join(self.root_dir, img_path.strip())
            if os.path.exists(full_path):
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            else:
                # Create a zero tensor if image doesn't exist
                images.append(torch.zeros((3, 224, 224)))
        
        # Stack all images into a single tensor
        images = torch.stack(images)
        
        return {
            'sensor_data': sensor_data,
            'images': images,
            'label': label
        } 