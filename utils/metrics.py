"""
Evaluation metrics and analysis utilities.

This module provides functions for model evaluation, metric computation,
and comprehensive analysis of prediction results.
"""

import numpy as np
import torch
from sklearn.metrics import r2_score

from .visualization import (
    plot_residuals, plot_error_analysis, plot_prediction_interval,
    plot_learning_curves_detailed
)


def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: PyTorch model to evaluate.
        data_loader: DataLoader for evaluation data.
        device: Device to run evaluation on.
        
    Returns:
        tuple: (targets, predictions, r2_score)
            - targets: Array of true target values.
            - predictions: Array of predicted values.
            - r2_score: R² coefficient of determination.
    """
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for sequences, bio_features, targets in data_loader:
            sequences, bio_features, targets = sequences.to(device), bio_features.to(device), targets.to(device)
            outputs = model(sequences, bio_features)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    r2 = r2_score(all_targets, all_predictions)

    return np.array(all_targets), np.array(all_predictions), r2


def plot_comprehensive_analysis(targets, predictions, train_losses, val_losses, base_save_path):
    """
    Generate comprehensive analysis plots and compute evaluation metrics.
    
    This function creates multiple diagnostic plots including:
    - Residuals vs predictions
    - Error distributions
    - Prediction intervals
    - Learning curves
    
    Args:
        targets: Array of true target values.
        predictions: Array of predicted values.
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        base_save_path: Base path for saving plots (without extension).
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - r2: R² score
            - rmse: Root mean square error
            - mae: Mean absolute error
            - mean_residual: Mean of residuals
            - std_residual: Standard deviation of residuals
            - prediction_interval_coverage: Coverage of 95% prediction intervals
    """
    # Compute metrics
    residuals = targets - predictions
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    print(f"Comprehensive Model Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean Residual: {np.mean(residuals):.4f}")
    print(f"Std Residual: {np.std(residuals):.4f}")
    
    # Generate plots
    plot_residuals(targets, predictions, f"{base_save_path}_residuals.png")
    plot_error_analysis(targets, predictions, f"{base_save_path}_error_analysis.png")
    coverage = plot_prediction_interval(targets, predictions, f"{base_save_path}_prediction_intervals.png")
    plot_learning_curves_detailed(train_losses, val_losses, f"{base_save_path}_detailed_learning_curves.png")
    
    print(f"Prediction interval coverage: {coverage*100:.2f}%")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'prediction_interval_coverage': coverage
    }
