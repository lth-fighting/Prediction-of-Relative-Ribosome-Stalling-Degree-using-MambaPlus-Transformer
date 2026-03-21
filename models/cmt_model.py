"""
Complete CNN-Mamba-Transformer (CMT) model for protein expression prediction.

This module integrates multi-scale CNN, bidirectional Mamba+, transformer,
and feature fusion into a unified architecture for sequence-based regression.
"""

import torch
import torch.nn as nn

from .cnn import MultiScaleCNN
from .mamba_blocks import BiMambaPlusEncoder
from .transformer import SequenceTransformer
from .fusion import FeatureFusion


class CNNMambaTransformer(nn.Module):
    """
    Complete CNN-Mamba-Transformer model for protein expression prediction.
    
    This model combines:
    1. Multi-scale CNN for local feature extraction from DNA sequences
    2. Bidirectional Mamba+ for efficient long-range dependency modeling
    3. Transformer for capturing global relationships
    4. Feature fusion to combine Mamba and Transformer representations
    5. Attention pooling for sequence summarization
    6. Fully connected layers for final regression
    
    Attributes:
        multi_scale_cnn (MultiScaleCNN): CNN feature extractor.
        mamba_branch (nn.Sequential): Mamba+ processing branch.
        transformer_branch (nn.Sequential): Transformer processing branch.
        feature_fusion (FeatureFusion): Fusion module.
        attention_pooling (nn.Sequential): Attention-based pooling.
        fc (nn.Sequential): Final regression layers.
    """
    
    def __init__(self, k: int = 3, num_bio_features: int = 7, 
                 cnn_out_channels: int = 64, mamba_hidden_size: int = 64, 
                 transformer_heads: int = 4, transformer_layers: int = 2,
                 fc1_size: int = 128, fc2_size: int = 64, dropout_rate: float = 0.15, 
                 fusion_method: str = 'weighted_sum'):
        """
        Initialize the CNN-Mamba-Transformer model.
        
        Args:
            k: K-mer length for DNA encoding.
            num_bio_features: Number of biophysical features.
            cnn_out_channels: Output channels per CNN branch.
            mamba_hidden_size: Hidden dimension for Mamba (not directly used, kept for compatibility).
            transformer_heads: Number of attention heads.
            transformer_layers: Number of transformer layers.
            fc1_size: Size of first fully connected layer.
            fc2_size: Size of second fully connected layer.
            dropout_rate: Dropout probability.
            fusion_method: Fusion strategy ('weighted_sum', 'concat', 'attention').
        """
        super(CNNMambaTransformer, self).__init__()

        in_channels = 4 ** k

        # Multi-scale CNN for local feature extraction
        self.multi_scale_cnn = MultiScaleCNN(
            in_channels=in_channels,
            cnn_out_channels=cnn_out_channels
        )

        cnn_output_size = cnn_out_channels * self.multi_scale_cnn.num_kernels

        # Bidirectional Mamba+ encoder branch
        self.mamba_branch = nn.Sequential(
            BiMambaPlusEncoder(
                d_model=cnn_output_size,
                d_state=32,
                d_conv=4,
                expand=2,
                dropout_rate=dropout_rate,
                num_layers=1
            ),
            nn.LayerNorm(cnn_output_size)
        )

        # Transformer branch
        self.transformer_branch = nn.Sequential(
            SequenceTransformer(
                d_model=cnn_output_size,
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dropout_rate=dropout_rate
            ),
            nn.LayerNorm(cnn_output_size)
        )

        # Feature fusion
        self.feature_fusion = FeatureFusion(
            d_model=cnn_output_size,
            fusion_method=fusion_method
        )

        # Attention-based pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(cnn_output_size, 1),
            nn.Softmax(dim=1)
        )

        # Final regression layers with biophysical features
        combined_features_size = cnn_output_size + num_bio_features
        self.fc = nn.Sequential(
            nn.Linear(combined_features_size, fc1_size),
            nn.LayerNorm(fc1_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(fc1_size, fc2_size),
            nn.LayerNorm(fc2_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(fc2_size, 1)
        )

    def forward(self, seq_input: torch.Tensor, 
                biophysical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN-Mamba-Transformer model.
        
        Args:
            seq_input: Encoded DNA sequence tensor of shape (batch, channels, seq_len).
            biophysical_features: Biophysical feature tensor of shape (batch, num_features).
            
        Returns:
            torch.Tensor: Predicted protein expression values of shape (batch,).
        """
        # CNN feature extraction
        cnn_out = self.multi_scale_cnn(seq_input)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len, features)

        # Parallel processing through Mamba+ and Transformer branches
        mamba_out = self.mamba_branch(cnn_out)
        transformer_out = self.transformer_branch(cnn_out)

        # Feature fusion
        fused_features = self.feature_fusion(mamba_out, transformer_out)

        # Attention-based pooling
        attention_weights = self.attention_pooling(fused_features)
        pooled = torch.sum(attention_weights * fused_features, dim=1)

        # Concatenate with biophysical features
        combined = torch.cat([pooled, biophysical_features], dim=-1)
        
        # Final regression
        output = self.fc(combined)

        return output.squeeze()
