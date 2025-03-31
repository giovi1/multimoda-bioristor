#!/usr/bin/env python
"""
Training script for the ResNet-LSTM multimodal model.
"""
import argparse
import os
from datetime import datetime

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data import BioristorDataset, create_data_loaders, save_test_indices
from src.models import ResNetLSTMConfig, MultimodalResNetLSTM
from src.trainers import ResNetLSTMTrainer
from src.utils import get_device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Multimodal ResNet-LSTM Model')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, default='data/mapped_data.csv',
                      help='Path to the CSV file with sensor data')
    parser.add_argument('--image_dir', type=str, default='data/images',
                      help='Directory containing the images')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=4,
                      help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--num_images', type=int, default=6,
                      help='Number of images per sample')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate')
    parser.add_argument('--resnet_type', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34'],
                      help='Type of ResNet to use')
    parser.add_argument('--freeze_resnet', action='store_true',
                      help='Freeze ResNet weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--track_features', action='store_true',
                      help='Track feature importance during training')
    
    # Save arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name for this experiment')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = get_device(args.device)
    print(f'Using device: {device}')
    
    # Create save directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset
    dataset = BioristorDataset(
        csv_file=args.data_csv,
        root_dir=args.image_dir
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader, test_indices = create_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Save test indices for later use
    save_test_indices(test_indices, save_dir)
    
    # Create model config
    config = ResNetLSTMConfig(
        num_classes=args.num_classes,
        image_size=args.image_size,
        num_images=args.num_images,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        resnet_type=args.resnet_type,
        freeze_resnet=args.freeze_resnet
    )
    
    # Create model
    model = MultimodalResNetLSTM(config).to(device)
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Create trainer
    trainer = ResNetLSTMTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )
    
    # Train the model
    if args.track_features:
        best_val_acc, feature_importance = trainer.train_with_feature_importance(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            patience=args.patience
        )
        
        # Save feature importance
        torch.save(feature_importance, os.path.join(save_dir, 'feature_importance.pt'))
        print(f"Feature importance saved to {os.path.join(save_dir, 'feature_importance.pt')}")
    else:
        best_val_acc = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            patience=args.patience
        )
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main() 