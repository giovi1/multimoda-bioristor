from src.trainers.trainer import Trainer
import torch

class ResNetLSTMTrainer(Trainer):
    """
    Trainer class for the ResNet-LSTM multimodal model.
    """
    def __init__(self, model, optimizer, scheduler, device, save_dir):
        """
        Initialize the trainer.
        
        Args:
            model: ResNet-LSTM model to train
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save model checkpoints
        """
        super().__init__(model, optimizer, scheduler, device, save_dir)
    
    def train_with_feature_importance(self, train_loader, val_loader, num_epochs, patience=10):
        """
        Train the model and track feature importance.
        
        This extended training method records the gradients of the embeddings
        to analyze which features are most important for predictions.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            patience: Early stopping patience
            
        Returns:
            best_val_acc: Best validation accuracy
            feature_importance: Dictionary with feature importance scores
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        # Track feature importance
        image_importance = torch.zeros(self.model.resnet_out_dim).to(self.device)
        sensor_importance = torch.zeros(self.model.lstm.hidden_size * 
                                        (2 if self.model.config.bidirectional else 1)).to(self.device)
        
        for epoch in range(num_epochs):
            # Train for one epoch
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                sensor_data = batch['sensor_data'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with feature tracking
                batch_size = images.size(0)
                
                # Process images with ResNet
                num_images = images.size(1)
                images_reshaped = images.reshape(-1, 3, self.model.config.image_size, self.model.config.image_size)
                image_features = self.model.resnet(images_reshaped)
                
                # Reshape back and average across images
                image_features = image_features.view(batch_size, num_images, -1)
                image_features = torch.mean(image_features, dim=1)
                
                # Process sensor data with LSTM
                sensor_data = sensor_data.unsqueeze(1)
                lstm_out, _ = self.model.lstm(sensor_data)
                sensor_features = lstm_out[:, -1, :]
                
                # Set requires_grad for feature importance tracking
                image_features.requires_grad_(True)
                sensor_features.requires_grad_(True)
                
                # Combine features and classify
                combined_features = torch.cat([image_features, sensor_features], dim=1)
                outputs = self.model.classifier(combined_features)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Accumulate gradients for feature importance
                if image_features.grad is not None:
                    image_importance += image_features.grad.abs().sum(dim=0)
                if sensor_features.grad is not None:
                    sensor_importance += sensor_features.grad.abs().sum(dim=0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print stats
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Check if this is the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc)
        
        # Normalize feature importance
        image_importance = image_importance / (epoch + 1)
        sensor_importance = sensor_importance / (epoch + 1)
        
        feature_importance = {
            'image_importance': image_importance.cpu().numpy(),
            'sensor_importance': sensor_importance.cpu().numpy(),
        }
        
        return best_val_acc, feature_importance 