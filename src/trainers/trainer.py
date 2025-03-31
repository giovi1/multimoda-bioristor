import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os

class Trainer:
    """
    Base trainer class for training and evaluating models.
    """
    def __init__(self, model, optimizer, scheduler, device, save_dir):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.criterion = CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            train_loss: Average training loss
            train_acc: Training accuracy
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            sensor_data = batch['sensor_data'].to(self.device)
            images = batch['images'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, sensor_data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        return train_loss, train_acc
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            val_loss: Average validation loss
            val_acc: Validation accuracy
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sensor_data = batch['sensor_data'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, sensor_data)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def train(self, train_loader, val_loader, num_epochs, patience=10):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            patience: Early stopping patience
            
        Returns:
            best_val_acc: Best validation accuracy
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
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
        
        return best_val_acc 