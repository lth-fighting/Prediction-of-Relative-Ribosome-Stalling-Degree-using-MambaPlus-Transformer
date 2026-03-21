"""
Dataset implementation for protein expression prediction.

This module defines the PyTorch Dataset class for handling DNA sequences,
biophysical features, and target protein expression values. It includes
data preprocessing, normalization, and encoding functionalities.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.encoding import DNA_kmer_onehot_encode


class ProteinExpressionDataset(Dataset):
    """
    PyTorch Dataset for protein expression prediction from DNA sequences.
    
    This dataset handles DNA sequence encoding, biophysical feature processing,
    and target variable normalization. It supports k-mer one-hot encoding for
    sequences and optional integration of biophysical features.
    
    Attributes:
        sequences (np.ndarray): Array of raw DNA sequences.
        target (np.ndarray): Normalized target expression values.
        features (np.ndarray): Normalized biophysical features.
        use_biophysical_features (bool): Whether to include biophysical features.
        k (int): K-mer length for sequence encoding.
    """
    
    def __init__(self, csv_file: str, target: str = 'Protein', 
                 encoding_method: str = 'kmer_onehot',
                 use_biophysical_features: bool = True, k: int = 3):
        """
        Initialize the protein expression dataset.
        
        Args:
            csv_file: Path to the CSV file containing sequence data.
            target: Name of the target column (default: 'Protein').
            encoding_method: Method for sequence encoding (currently supports 'kmer_onehot').
            use_biophysical_features: Whether to include biophysical features.
            k: Length of k-mers for sequence encoding.
        """
        self.data = pd.read_csv(csv_file)
        self.sequences = self.data['Sequence'].values
        self.target = target
        self.use_biophysical_features = use_biophysical_features
        self.k = k

        # Biophysical features preprocessing
        if use_biophysical_features:
            biophysical_features_cols = [
                'cdsCAI', 'utrCdsStructureMFE', 'fivepCdsStructureMFE', 
                'threepCdsStructureMFE', 'cdsBottleneckPosition', 
                'cdsNucleotideContentAT', 'cdsHydropathyIndex'
            ]
            self.features = self.data[biophysical_features_cols].values
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)

        # Target variable normalization (z-score normalization)
        self.data['cdsBottleneckRelativeStrength_normalized'] = (
            self.data[target] - self.data[target].mean()
        ) / self.data[target].std()
        self.target = self.data['cdsBottleneckRelativeStrength_normalized'].values

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            tuple: (sequence_tensor, features_tensor, target_tensor)
                - sequence_tensor: Encoded DNA sequence (C x L)
                - features_tensor: Biophysical features (7-dim vector)
                - target_tensor: Normalized target value
        """
        sequence = self.sequences[index]
        sequence_encoded = DNA_kmer_onehot_encode(sequence, self.k)
        sequence_tensor = torch.tensor(sequence_encoded, dtype=torch.float32).transpose(0, 1)

        target = torch.tensor(self.target[index], dtype=torch.float32)
        features = torch.tensor(self.features[index], dtype=torch.float32)

        return sequence_tensor, features, target
