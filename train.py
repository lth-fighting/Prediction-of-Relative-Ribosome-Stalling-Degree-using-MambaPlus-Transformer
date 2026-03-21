"""
Training utilities for the CNN-Mamba-Transformer model.

This module provides the main training loop with checkpointing support,
early stopping, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.visualization import plot_training_progress, plot_prediction_interval
from utils.metrics import evaluate_model


def train_model_with_checkpoints(model, train_loader, val_loader, criterion, optimizer, 
                               scheduler, num_epochs, initial_epoch=0, patience=10,
                               checkpoint_freq=1, resume_from=None, checkpoint_dir=None,
                               plot_dir=None, device=None):
    """
    Train the model with checkpoint support.
    
    This function implements the complete training loop with:
    - Automatic checkpoint saving at specified intervals
    - Early stopping based on validation loss
    - Best model tracking
    - Training progress visualization
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        num_epochs: Total number of training epochs.
        initial_epoch: Starting epoch number (for resuming).
        patience: Number of epochs to wait for improvement before early stopping.
        checkpoint_freq: Frequency of checkpoint saving (in epochs).
        resume_from: Path to checkpoint to resume from.
        checkpoint_dir: Directory for saving checkpoints.
        plot_dir: Directory for saving training plots.
        device: Device to run training on.
        
    Returns:
        tuple: (trained_model, train_losses, val_losses)
    """
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    # Resume from checkpoint if specified
    if resume_from:
        epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, resume_from, device
        )
        initial_epoch = epoch + 1
        print(f"Resuming training from epoch {initial_epoch}")
    
    for epoch in range(initial_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for sequence, bio_features, target in train_loader:
            sequence, bio_features, target = sequence.to(device), bio_features.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequence, bio_features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequence.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequence, bio_features, target in val_loader:
                sequence, bio_features, target = sequence.to(device), bio_features.to(device), target.to(device)
                
                outputs = model(sequence, bio_features)
                loss = criterion(outputs, target)
                val_loss += loss.item() * sequence.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'best_val_loss': best_val_loss,
                'timestamp': datetime.now().isoformat()
            }, best_model_path)
            print(f"New best model saved: {best_model_path}")
        else:
            epochs_no_improve += 1

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses, 
                          val_losses, best_val_loss, 
                          f"checkpoint_epoch_{epoch+1}.pth", checkpoint_dir)
            
            # Generate training progress plot
            plot_training_progress(train_losses, val_losses, epoch, plot_dir)
            
            # Evaluate on validation set and generate prediction plot
            val_targets, val_predictions, val_r2 = evaluate_model(model, val_loader, device)
            plt.figure(figsize=(8, 6))
            plt.scatter(val_targets, val_predictions, alpha=0.5)
            plt.plot([val_targets.min(), val_targets.max()],
                     [val_targets.min(), val_targets.max()], 'r--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Validation Predictions - R² = {val_r2:.4f}')
            plot_path = os.path.join(plot_dir, f'validation_predictions_epoch_{epoch+1}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Validation prediction plot saved: {plot_path}")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses
