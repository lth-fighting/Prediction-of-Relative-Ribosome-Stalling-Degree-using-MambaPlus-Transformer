"""
Checkpoint management utilities for model training.

This module provides functions for saving and loading training checkpoints,
allowing resumption of training from interrupted sessions.
"""

import os
import torch
from datetime import datetime

# Global checkpoint and plot directories (will be set by main script)
CHECKPOINT_DIR = "./checkpoints"
PLOT_DIR = "./training_plots"


def set_checkpoint_dirs(checkpoint_dir: str, plot_dir: str):
    """
    Set global checkpoint and plot directories.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints.
        plot_dir: Directory for saving training plots.
    """
    global CHECKPOINT_DIR, PLOT_DIR
    CHECKPOINT_DIR = checkpoint_dir
    PLOT_DIR = plot_dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
                   best_val_loss, filename=None, checkpoint_dir=None):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
        train_losses: List of training losses.
        val_losses: List of validation losses.
        best_val_loss: Best validation loss achieved.
        filename: Optional custom filename.
        checkpoint_dir: Optional custom checkpoint directory.
        
    Returns:
        str: Path to the saved checkpoint.
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load training checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the checkpoint on.
        
    Returns:
        tuple: (epoch, train_losses, val_losses, best_val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return (checkpoint['epoch'], checkpoint['train_losses'], 
            checkpoint['val_losses'], checkpoint['best_val_loss'])


def find_latest_checkpoint(checkpoint_dir=None):
    """
    Find the most recent checkpoint file.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints.
        
    Returns:
        str or None: Path to the latest checkpoint, or None if no checkpoints found.
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Sort by modification time, return latest
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), 
                         reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0])
