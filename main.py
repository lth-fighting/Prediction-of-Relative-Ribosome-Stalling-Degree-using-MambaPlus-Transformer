"""
Main script for protein expression prediction using CNN-Mamba-Transformer.

This script orchestrates the complete workflow including:
- Data loading and preprocessing
- Model initialization
- Training with checkpoint support
- Evaluation on test set
- Comprehensive analysis and visualization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Import project modules
from data.dataset import ProteinExpressionDataset
from models.cmt_model import CNNMambaTransformer
from train import train_model_with_checkpoints
from utils.metrics import evaluate_model, plot_comprehensive_analysis
from utils.checkpoint import set_checkpoint_dirs, find_latest_checkpoint
from config import ModelConfig, TrainingConfig, DeviceConfig


def main():
    """
    Main execution function for protein expression prediction.
    
    This function sets up the configuration, loads data, initializes the model,
    runs training, and performs evaluation with comprehensive analysis.
    """
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    device_config = DeviceConfig()
    
    # Set up directories
    set_checkpoint_dirs(training_config.checkpoint_dir, training_config.plot_dir)
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Device setup
    device = torch.device(device_config.device_name)
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ProteinExpressionDataset(
        training_config.train_data_path, 
        k=model_config.k
    )
    val_dataset = ProteinExpressionDataset(
        training_config.val_data_path, 
        k=model_config.k
    )
    test_dataset = ProteinExpressionDataset(
        training_config.test_data_path, 
        k=model_config.k
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = CNNMambaTransformer(
        k=model_config.k,
        num_bio_features=model_config.num_bio_features,
        cnn_out_channels=model_config.cnn_out_channels,
        mamba_hidden_size=model_config.mamba_hidden_size,
        transformer_heads=model_config.transformer_heads,
        transformer_layers=model_config.transformer_layers,
        fc1_size=model_config.fc1_size,
        fc2_size=model_config.fc2_size,
        dropout_rate=model_config.dropout_rate,
        fusion_method=model_config.fusion_method
    ).to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config.learning_rate, 
        weight_decay=training_config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=training_config.scheduler_patience, 
        factor=training_config.scheduler_factor
    )
    
    # Check for existing checkpoint
    latest_checkpoint = find_latest_checkpoint(training_config.checkpoint_dir)
    resume_training = False
    
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        user_input = input("Resume training from checkpoint? (y/n): ")
        if user_input.lower() == 'y':
            resume_training = True
    
    # Train model
    print("Starting training...")
    model, train_losses, val_losses = train_model_with_checkpoints(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=training_config.num_epochs,
        patience=training_config.patience,
        checkpoint_freq=training_config.checkpoint_freq,
        resume_from=latest_checkpoint if resume_training else None,
        checkpoint_dir=training_config.checkpoint_dir,
        plot_dir=training_config.plot_dir,
        device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(training_config.output_dir, 
                             f'{model_config.fusion_method}_training_curve.png'))
    plt.close()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_targets, test_predictions, test_r2 = evaluate_model(model, test_loader, device)
    print(f'Test R²: {test_r2:.4f}')
    
    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets, test_predictions, alpha=0.5)
    plt.plot([test_targets.min(), test_targets.max()],
             [test_targets.min(), test_targets.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True vs Predicted Values - R² = {test_r2:.4f}')
    plt.savefig(os.path.join(training_config.output_dir, 
                            f'{model_config.fusion_method}_predictions_vs_true.png'))
    plt.close()
    
    # Comprehensive analysis
    print("Generating comprehensive analysis...")
    metrics = plot_comprehensive_analysis(
        test_targets, 
        test_predictions, 
        train_losses, 
        val_losses, 
        os.path.join(training_config.output_dir, 'comprehensive_analysis')
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(training_config.output_dir, 'comprehensive_analysis.csv'), 
                      index=False)
    print("Comprehensive evaluation metrics saved to CSV file")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(training_config.output_dir, 'final_model.pth'))
    print("Final model saved to 'final_model.pth'")
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved to: {training_config.output_dir}")


if __name__ == '__main__':
    main()
