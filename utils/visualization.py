"""
Visualization utilities for model training and evaluation.

This module provides functions for generating various diagnostic plots
including loss curves, prediction scatter plots, residual analysis,
and error distributions.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals(targets, predictions, save_path):
    """
    Plot residuals analysis.
    
    Args:
        targets: Array of true target values.
        predictions: Array of predicted values.
        save_path: Path to save the plot.
    """
    residuals = targets - predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs predicted values
    axes[0].scatter(predictions, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_analysis(targets, predictions, save_path):
    """
    Plot comprehensive error analysis.
    
    Args:
        targets: Array of true target values.
        predictions: Array of predicted values.
        save_path: Path to save the plot.
    """
    errors = np.abs(targets - predictions)
    relative_errors = np.abs((targets - predictions) / (targets + 1e-8))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Absolute error distribution
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Absolute Errors')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative error distribution
    axes[0, 1].hist(relative_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Relative Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Relative Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error vs true values
    axes[1, 0].scatter(targets, errors, alpha=0.5, color='green')
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Absolute Error vs True Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error vs predicted values
    axes[1, 1].scatter(predictions, errors, alpha=0.5, color='purple')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error vs Predicted Values')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_interval(targets, predictions, save_path, confidence=0.95):
    """
    Plot prediction intervals.
    
    Args:
        targets: Array of true target values.
        predictions: Array of predicted values.
        save_path: Path to save the plot.
        confidence: Confidence level for prediction intervals.
        
    Returns:
        float: Coverage proportion within prediction intervals.
    """
    residuals = targets - predictions
    std_residual = np.std(residuals)
    
    # Compute prediction intervals (assuming normal distribution)
    z_score = 1.96  # 95% confidence interval
    lower_bound = predictions - z_score * std_residual
    upper_bound = predictions + z_score * std_residual
    
    # Calculate coverage
    in_interval = np.sum((targets >= lower_bound) & (targets <= upper_bound)) / len(targets)
    
    plt.figure(figsize=(12, 8))
    
    # Sort for better visualization
    sort_idx = np.argsort(targets)
    targets_sorted = targets[sort_idx]
    predictions_sorted = predictions[sort_idx]
    lower_sorted = lower_bound[sort_idx]
    upper_sorted = upper_bound[sort_idx]
    
    plt.plot(targets_sorted, targets_sorted, 'r--', alpha=0.8, label='Perfect Prediction')
    plt.scatter(targets, predictions, alpha=0.6, label='Predictions', s=20)
    plt.fill_between(targets_sorted, lower_sorted, upper_sorted, alpha=0.3, 
                    color='orange', label=f'{confidence*100:.0f}% Prediction Interval')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Prediction Intervals (Coverage: {in_interval*100:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return in_interval


def plot_learning_curves_detailed(train_losses, val_losses, save_path):
    """
    Plot detailed learning curves including loss ratio analysis.
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(15, 5))
    
    # Original loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log-scale loss curves
    plt.subplot(1, 3, 2)
    plt.semilogy(train_losses, label='Training Loss', linewidth=2)
    plt.semilogy(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Curves (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss ratio (indicator of overfitting)
    plt.subplot(1, 3, 3)
    loss_ratio = [val/train if train > 0 else 0 for train, val in zip(train_losses, val_losses)]
    plt.plot(loss_ratio, linewidth=2, color='purple')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss / Train Loss')
    plt.title('Loss Ratio (Indication of Overfitting)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_progress(train_losses, val_losses, epoch, save_dir):
    """
    Plot training progress with recent epochs detail.
    
    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        epoch: Current epoch number.
        save_dir: Directory to save the plot.
    """
    plt.figure(figsize=(12, 5))
    
    # Full loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Progress (Epoch {epoch + 1})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recent epochs detail
    plt.subplot(1, 2, 2)
    recent_epochs = min(20, len(train_losses))
    if recent_epochs > 0:
        start_idx = len(train_losses) - recent_epochs
        plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], 
                label='Training Loss', linewidth=2, marker='o')
        plt.plot(range(start_idx, len(train_losses)), val_losses[start_idx:], 
                label='Validation Loss', linewidth=2, marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Recent {recent_epochs} Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_progress_epoch_{epoch+1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved: {plot_path}")
