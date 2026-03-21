"""
Utility functions module.

This module provides helper functions for encoding, metrics computation,
visualization, and checkpoint management.
"""

from .encoding import DNA_kmer_onehot_encode
from .metrics import evaluate_model, plot_comprehensive_analysis
from .visualization import (
    plot_residuals, plot_error_analysis, plot_prediction_interval,
    plot_learning_curves_detailed, plot_training_progress
)
from .checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

__all__ = [
    'DNA_kmer_onehot_encode',
    'evaluate_model',
    'plot_comprehensive_analysis',
    'plot_residuals',
    'plot_error_analysis',
    'plot_prediction_interval',
    'plot_learning_curves_detailed',
    'plot_training_progress',
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint'
]
