# Prediction-of-Relative-Ribosome-Stalling-Degree-using-MambaPlus-Transformer
A hybrid architecture where a multi-scale CNN first encodes DNA k-mers, then parallel Bi-MambaPlus and Transformer branches capture both directional and global sequence patterns. Their outputs are fused and combined with seven biophysical features before final regression layers predict ribosome stalling.

# CNN-MambaPlus-Transformer for Ribosome Stalling Prediction

## Overview
This project implements a hybrid deep learning model for predicting the relative degree of ribosome stalling. The model integrates DNA sequence information with seven biophysical features using a multi‑scale CNN, bidirectional MambaPlus, and Transformer encoder. It supports checkpoint recovery, comprehensive visualization, and evaluation metrics.

## Architecture

The model (`CNNMambaTransformer`) consists of several key components:

### 1. Multi‑Scale CNN
- **Input**: DNA sequence encoded as k‑mer one‑hot vectors (k=3, dimension = 4³ = 64).
- **Parallel Convolutions**: Three branches with kernel sizes 3, 5, and 7. Each branch applies `Conv1d → BatchNorm → ReLU → Dropout`.
- **Output**: Concatenated along channel dimension → 192‑dimensional feature maps per position.

### 2. Bi‑MambaPlus Encoder
- **Bidirectional Processing**: Forward and backward MambaPlus blocks, each containing:
  - A depthwise convolution (kernel size 3) with SiLU activation.
  - A Mamba state‑space model (SSM) block.
  - A gating mechanism that combines SSM output and convolved features via a learnable sigmoid gate.
  - Residual connections and layer normalization.
- **Output**: Sum of forward and backward outputs, passed through an FFN (linear → GELU → dropout → linear) with layer normalization.

### 3. Sequence Transformer
- **Positional Encoding**: Learnable positional embeddings.
- **Transformer Blocks**: Multi‑head self‑attention (4 heads), followed by feed‑forward network (dimension 4×d_model) with GELU activation.
- **Layer Norm & Dropout**: Applied after each sub‑layer.

### 4. Feature Fusion
- Takes outputs from the Bi‑MambaPlus and Transformer branches.
- Supports three fusion methods (configurable):
  - **weighted_sum**: Learnable weights α and β normalized via sigmoid.
  - **concat**: Concatenation followed by linear projection.
  - **attention**: Multi‑head attention over concatenated features followed by projection.
- Default: `weighted_sum`.

### 5. Attention Pooling
- Computes attention weights over the fused sequence features using a linear layer and softmax.
- Outputs a fixed‑length vector (weighted sum across positions).

### 6. Regression Head
- Concatenates the pooled sequence vector with 7 biophysical features (scaled via StandardScaler).
- Two fully connected layers (128 → 64) with GELU, LayerNorm, and Dropout.
- Final linear layer outputs a scalar prediction.

### Data Flow
```
DNA Sequence → k‑mer Encoding → Multi‑Scale CNN → (Bi‑MambaPlus + Transformer) → Feature Fusion → Attention Pooling → Concatenate with Biophysical Features → FC Layers → Prediction
```

## Dependencies
- Python ≥ 3.8
- PyTorch ≥ 2.0
- `mamba-ssm` (install separately: [Mamba GitHub](https://github.com/state-spaces/mamba))
- pandas, numpy, scikit‑learn, matplotlib

Installation example:
```bash
pip install torch pandas numpy scikit-learn matplotlib
# Follow mamba-ssm installation instructions (requires CUDA and appropriate compiler)
```

## Data Preparation

### Input CSV Format
Three CSV files are required for training, validation, and testing (paths hardcoded in `ProteinExpressionDataset`). Each must contain the following columns:

| Column | Description |
|--------|-------------|
| `Sequence` | DNA sequence string |
| `Protein` | Raw target value (normalized internally) |
| `cdsCAI` | Codon Adaptation Index |
| `utrCdsStructureMFE` | UTR‑CDS structure minimum free energy |
| `fivepCdsStructureMFE` | 5′ CDS structure MFE |
| `threepCdsStructureMFE` | 3′ CDS structure MFE |
| `cdsBottleneckPosition` | Bottleneck position |
| `cdsNucleotideContentAT` | AT content in CDS |
| `cdsHydropathyIndex` | Hydropathy index |

The target is automatically Z‑score normalized.

### File Locations
- `./processed_data/train_Ecoli_data.csv`
- `./processed_data/val_Ecoli_data.csv`
- `./processed_data/test_Ecoli_data.csv`

Modify these paths in the code if necessary.

## Usage

### Train from Scratch
Run the script directly:
```bash
python cmt.py
```

Default settings:
- Batch size: 64
- Epochs: 100
- Learning rate: 1e‑3
- Optimizer: AdamW with weight decay 1e‑5
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping patience: 10

### Resume Training
Checkpoints are saved in `./checkpoints/` every epoch. When restarting, the script will prompt whether to resume from the latest checkpoint. Enter `y` to continue.

### Outputs
After training, the following files are generated in the current directory:
- **`weighted_sum_training_curve_me_mc_mamtransformer.png`** – training/validation loss curves
- **`weighted_sum_predictions_vs_true_me_mc_mamtransformer.png`** – scatter plot of predictions vs. true values (test set)
- **`comprehensive_analysis.png`** – residuals vs. predicted
- **`comprehensive_analysis_error_analysis.png`** – error distribution plots
- **`comprehensive_analysis_prediction_intervals.png`** – 95% prediction intervals
- **`comprehensive_analysis_detailed_learning_curves.png`** – detailed learning curves (loss ratio, log scale)
- **`comprehensive_analysis.csv`** – evaluation metrics (R², RMSE, MAE, etc.)
- **`best_model.pth`** – best model weights (lowest validation loss)

Checkpoints and training plots are saved in `./checkpoints/` and `./training_plots/`.

### Hyperparameter Tuning
Modify the following in `__main__`:
```python
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
K = 3                     # k‑mer length
CNN_OUT_CHANNELS = 64
MAMBA_HIDDEN_SIZE = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
WEIGHT_DECAY = 1e-5
```

## Evaluation Metrics
The `plot_comprehensive_analysis` function computes:
- R² score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean and standard deviation of residuals
- Prediction interval coverage (95% CI)

## Citation
If you use this code, please cite the relevant publication (if any) or acknowledge the repository.

---

For questions or issues, please open an issue on GitHub.
