"""
Configuration settings for protein expression prediction model.

This module contains all hyperparameters and configuration settings for
training and evaluating the CNN-Mamba-Transformer model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the CNN-Mamba-Transformer model."""
    
    # Model architecture
    k: int = 3  # K-mer length
    num_bio_features: int = 7  # Number of biophysical features
    cnn_out_channels: int = 64  # CNN output channels per branch
    mamba_hidden_size: int = 64  # Mamba hidden dimension (for compatibility)
    transformer_heads: int = 4  # Number of attention heads
    transformer_layers: int = 2  # Number of transformer layers
    fc1_size: int = 128  # First fully connected layer size
    fc2_size: int = 64  # Second fully connected layer size
    dropout_rate: float = 0.15  # Dropout probability
    fusion_method: str = 'weighted_sum'  # Feature fusion method
    
    # Mamba specific parameters
    d_state: int = 32  # SSM state dimension
    d_conv: int = 4  # SSM convolution dimension
    expand: int = 2  # Expansion factor


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 10  # Number of epochs to wait for improvement
    
    # Scheduler
    scheduler_patience: int = 3  # Patience for ReduceLROnPlateau
    scheduler_factor: float = 0.5  # Factor to reduce learning rate
    
    # Checkpoint
    checkpoint_freq: int = 1  # Save checkpoint every N epochs
    
    # Paths
    train_data_path: str = './processed_data/train_Ecoli_data.csv'
    val_data_path: str = './processed_data/val_Ecoli_data.csv'
    test_data_path: str = './processed_data/test_Ecoli_data.csv'
    checkpoint_dir: str = './checkpoints'
    plot_dir: str = './training_plots'
    output_dir: str = '.'


@dataclass
class DeviceConfig:
    """Configuration for device selection."""
    
    use_cuda: bool = True
    cuda_device: Optional[int] = None  # Specific GPU device ID
    
    @property
    def device_name(self) -> str:
        """Get the device name for PyTorch."""
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_device is not None:
                return f"cuda:{self.cuda_device}"
            return "cuda"
        return "cpu"
