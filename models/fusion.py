"""
Feature fusion module for combining Mamba and Transformer representations.

This module implements various fusion strategies to combine features from
different branches of the model architecture.
"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Flexible feature fusion module supporting multiple fusion strategies.
    
    This module combines features from two branches (typically Mamba and
    Transformer) using one of several fusion methods: weighted sum,
    concatenation, or attention-based fusion.
    
    Attributes:
        fusion_method (str): Method used for feature fusion.
        mamba_weight (nn.Parameter): Learnable weight for Mamba branch (weighted_sum).
        transformer_weight (nn.Parameter): Learnable weight for Transformer branch (weighted_sum).
        fusion_proj (nn.Linear): Projection layer for concat/attention methods.
        attention (nn.MultiheadAttention): Attention mechanism for attention fusion.
    """
    
    def __init__(self, d_model: int, fusion_method: str = 'weighted_sum'):
        """
        Initialize feature fusion module.
        
        Args:
            d_model: Dimension of input features.
            fusion_method: Fusion strategy. Options: 'weighted_sum', 'concat', 'attention'.
        """
        super().__init__()

        self.fusion_method = fusion_method

        if self.fusion_method == 'weighted_sum':
            self.mamba_weight = nn.Parameter(torch.tensor(0.5))
            self.transformer_weight = nn.Parameter(torch.tensor(0.5))

        elif self.fusion_method == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
            
        elif self.fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model * 2,
                num_heads=4,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, mamba_features: torch.Tensor, 
                transformer_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from Mamba and Transformer branches.
        
        Args:
            mamba_features: Features from Mamba branch (batch, seq_len, d_model).
            transformer_features: Features from Transformer branch (batch, seq_len, d_model).
            
        Returns:
            torch.Tensor: Fused features of shape (batch, seq_len, d_model).
        """
        if self.fusion_method == 'weighted_sum':
            # Softmax normalization of weights
            alpha = torch.sigmoid(self.mamba_weight)
            beta = torch.sigmoid(self.transformer_weight)
            total = alpha + beta
            fused = (alpha / total) * mamba_features + (beta / total) * transformer_features

        elif self.fusion_method == 'concat':
            fused = torch.cat([mamba_features, transformer_features], dim=-1)
            fused = self.fusion_proj(fused)

        elif self.fusion_method == 'attention':
            combined = torch.cat([mamba_features, transformer_features], dim=-1)
            attented, _ = self.attention(combined, combined, combined)
            fused = self.fusion_proj(attented)

        return fused
