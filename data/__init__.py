"""
Data module for protein expression prediction.

This module provides dataset classes and data loading utilities for processing
DNA sequences and biophysical features for protein expression prediction tasks.
"""

from .dataset import ProteinExpressionDataset

__all__ = ['ProteinExpressionDataset']
