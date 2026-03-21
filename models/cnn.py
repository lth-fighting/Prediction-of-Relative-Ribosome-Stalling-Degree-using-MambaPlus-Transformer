"""
Multi-scale CNN module for feature extraction from DNA sequences.

This module implements a multi-scale convolutional neural network that extracts
features from encoded DNA sequences using multiple kernel sizes to capture
patterns at different resolutions.
"""

import torch
import torch.nn as nn
from typing import List


class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D Convolutional Neural Network for sequence feature extraction.
    
    This module applies multiple parallel 1D convolutions with different kernel
    sizes to capture features at various scales. Each branch consists of
    convolution, batch normalization, ReLU activation, and dropout.
    
    Attributes:
        num_kernels (int): Number of kernel sizes used.
        conv_branches (nn.ModuleList): List of convolutional branches.
    """
    
    def __init__(self, in_channels: int, cnn_out_channels: int = 64, 
                 kernel_sizes: List[int] = [3, 5, 7], dropout_rate: float = 0.2):
        """
        Initialize the multi-scale CNN module.
        
        Args:
            in_channels: Number of input channels (k-mer vocabulary size).
            cnn_out_channels: Number of output channels per convolutional branch.
            kernel_sizes: List of kernel sizes for each branch.
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        self.num_kernels = len(kernel_sizes)

        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, cnn_out_channels, k, padding=k//2),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-scale CNN.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len).
            
        Returns:
            torch.Tensor: Concatenated outputs from all branches,
                         shape (batch_size, cnn_out_channels * num_kernels, seq_len).
        """
        branch_outputs = [branch(x) for branch in self.conv_branches]
        return torch.cat(branch_outputs, dim=1)
