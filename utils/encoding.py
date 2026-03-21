"""
DNA sequence encoding utilities.

This module provides functions for encoding DNA sequences into numerical
representations suitable for deep learning models.
"""

import numpy as np
from itertools import product


def DNA_kmer_onehot_encode(sequence: str, k: int = 2) -> np.ndarray:
    """
    Encode DNA sequence using k-mer one-hot encoding.
    
    This function converts a DNA sequence into a one-hot encoded representation
    of k-mers. Each k-mer is mapped to a unique character, and then to a
    one-hot vector. Unknown k-mers are encoded as -1 vectors.
    
    Args:
        sequence: Input DNA sequence string.
        k: Length of k-mers (default: 2).
        
    Returns:
        np.ndarray: One-hot encoded representation of shape (num_kmers, vocab_size).
        
    Raises:
        ValueError: If the vocabulary size exceeds the available character set.
    """
    kmer_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    base = ['A', 'T', 'C', 'G']
    all_kmer = [''.join(p) for p in product(base, repeat=k)]

    # Check if vocabulary fits in available characters
    if len(all_kmer) > len(kmer_chars):
        raise ValueError(f"Vocabulary size ({len(all_kmer)}) exceeds available characters ({len(kmer_chars)})")

    # Map each k-mer to a character
    kmer_to_char = {}
    for i, kmer in enumerate(all_kmer[:len(kmer_chars)]):
        kmer_to_char[kmer] = kmer_chars[i]

    # Convert sequence to k-mers
    kmer_sequence = []
    step = 1
    for i in range(0, len(sequence) - k + 1, step):
        kmer = sequence[i:i+k].upper()
        char = kmer_to_char.get(kmer, 'N')
        kmer_sequence.append(char)

    # Create one-hot mapping
    char_mapping = {}
    for i, char in enumerate(kmer_chars[:len(kmer_to_char)]):
        one_hot = [0] * len(kmer_to_char)
        one_hot[i] = 1
        char_mapping[char] = one_hot
    
    # Special handling for unknown characters
    default = [-1] * len(kmer_to_char)
    encoded = np.array([char_mapping.get(char, default) for char in kmer_sequence])

    return encoded
