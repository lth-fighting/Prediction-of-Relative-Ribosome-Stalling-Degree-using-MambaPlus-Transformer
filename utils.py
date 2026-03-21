"""
utils.py

提供DNA序列编码、数据集加载、检查点保存/加载以及可视化工具。
所有功能均保持原有逻辑，添加了详细的文档说明。
"""

import os
import json
from datetime import datetime
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 检查点与绘图目录
CHECKPOINT_DIR = "./checkpoints"
PLOT_DIR = "./training_plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def DNA_kmer_onehot_encode(sequence: str, k: int = 2) -> np.ndarray:
    """
    将DNA序列编码为k-mer one-hot向量。

    参数:
        sequence (str): DNA序列字符串。
        k (int): k-mer长度。

    返回:
        np.ndarray: 形状为 (L, 4^k) 的one-hot编码矩阵，其中L为k-mer的数量。
    """
    kmer_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    base = ['A', 'T', 'C', 'G']
    all_kmer = [''.join(p) for p in product(base, repeat=k)]

    kmer_to_char = {}
    for i, kmer in enumerate(all_kmer[:len(kmer_chars)]):
        kmer_to_char[kmer] = kmer_chars[i]

    kmer_sequence = []
    step = 1
    for i in range(0, len(sequence) - k + 1, step):
        kmer = sequence[i:i+k].upper()
        char = kmer_to_char.get(kmer, 'N')
        kmer_sequence.append(char)

    char_mapping = {}
    for i, char in enumerate(kmer_chars[:len(kmer_to_char)]):
        one_hot = [0] * len(kmer_to_char)
        one_hot[i] = 1
        if char == 'N':
            char_mapping[char] = [-1] * len(kmer_to_char)
        else:
            char_mapping[char] = one_hot

    default = [-1] * len(kmer_to_char)
    encoded = np.array([char_mapping.get(char, default) for char in kmer_sequence])
    return encoded


class ProteinExpressionDataset(Dataset):
    """
    蛋白质表达数据集类，支持DNA序列和生物物理特征。

    参数:
        csv_file (str): CSV文件路径。
        target (str): 目标列名，默认为'Protein'。
        encoding_method (str): 编码方法，目前仅支持'kmer_onehot'。
        use_biophysical_features (bool): 是否使用生物物理特征。
        k (int): k-mer长度。
    """
    def __init__(self, csv_file: str, target: str = 'Protein', encoding_method: str = 'kmer_onehot',
                 use_biophysical_features: bool = True, k: int = 3):
        self.data = pd.read_csv(csv_file)
        self.sequences = self.data['Sequence'].values
        self.target = target
        self.use_biophysical_features = use_biophysical_features
        self.k = k

        if use_biophysical_features:
            biophysical_features_cols = ['cdsCAI', 'utrCdsStructureMFE', 'fivepCdsStructureMFE', 'threepCdsStructureMFE',
                                         'cdsBottleneckPosition', 'cdsNucleotideContentAT', 'cdsHydropathyIndex']
            self.features = self.data[biophysical_features_cols].values
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)

        self.data['cdsBottleneckRelativeStrength_normalized'] = (self.data[target] - self.data[target].mean()) / self.data[target].std()
        self.target = self.data['cdsBottleneckRelativeStrength_normalized'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        sequence_encoded = DNA_kmer_onehot_encode(sequence, self.k)
        sequence_tensor = torch.tensor(sequence_encoded, dtype=torch.float32).transpose(0, 1)

        target = torch.tensor(self.target[index], dtype=torch.float32)

        features = torch.tensor(self.features[index], dtype=torch.float32)

        return sequence_tensor, features, target


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses,
                   best_val_loss, filename=None):
    """保存训练检查点"""
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载训练检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"从检查点恢复训练: {checkpoint_path}")
    print(f"从第 {checkpoint['epoch']} 轮继续训练")
    print(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")

    return (checkpoint['epoch'], checkpoint['train_losses'],
            checkpoint['val_losses'], checkpoint['best_val_loss'])


def find_latest_checkpoint():
    """找到最新的检查点文件"""
    if not os.path.exists(CHECKPOINT_DIR):
        return None

    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)
    return os.path.join(CHECKPOINT_DIR, checkpoint_files[0])


def plot_training_progress(train_losses, val_losses, epoch, save_dir=PLOT_DIR):
    """绘制训练进度图（损失曲线及最近20轮细节）"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Progress (Epoch {epoch})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最近20个epoch的详细视图
    plt.subplot(1, 2, 2)
    recent_epochs = min(20, len(train_losses))
    if recent_epochs > 0:
        start_idx = len(train_losses) - recent_epochs
        plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:],
                label='Training Loss', linewidth=2, marker='o')
        plt.plot(range(start_idx, len(train_losses)), val_losses[start_idx:],
                label='Validation Loss', linewidth=2, marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Recent {recent_epochs} Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_progress_epoch_{epoch+1}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练进度图已保存: {plot_path}")


def plot_residuals(targets, predictions, save_path):
    """绘制残差图"""
    residuals = targets - predictions

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].scatter(predictions, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_analysis(targets, predictions, save_path):
    """绘制误差分析图（绝对误差、相对误差分布等）"""
    errors = np.abs(targets - predictions)
    relative_errors = np.abs((targets - predictions) / (targets + 1e-8))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Absolute Errors')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(relative_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Relative Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Relative Errors')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(targets, errors, alpha=0.5, color='green')
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Absolute Error vs True Values')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(predictions, errors, alpha=0.5, color='purple')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error vs Predicted Values')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_interval(targets, predictions, save_path, confidence=0.95):
    """绘制预测区间图"""
    residuals = targets - predictions
    std_residual = np.std(residuals)

    z_score = 1.96  # 95%置信区间
    lower_bound = predictions - z_score * std_residual
    upper_bound = predictions + z_score * std_residual

    in_interval = np.sum((targets >= lower_bound) & (targets <= upper_bound)) / len(targets)

    plt.figure(figsize=(12, 8))

    sort_idx = np.argsort(targets)
    targets_sorted = targets[sort_idx]
    predictions_sorted = predictions[sort_idx]
    lower_sorted = lower_bound[sort_idx]
    upper_sorted = upper_bound[sort_idx]

    plt.plot(targets_sorted, targets_sorted, 'r--', alpha=0.8, label='Perfect Prediction')
    plt.scatter(targets, predictions, alpha=0.6, label='Predictions', s=20)
    plt.fill_between(targets_sorted, lower_sorted, upper_sorted, alpha=0.3,
                    color='orange', label=f'{confidence*100:.0f}% Prediction Interval')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Prediction Intervals (Coverage: {in_interval*100:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return in_interval


def plot_learning_curves_detailed(train_losses, val_losses, save_path):
    """绘制详细的学习曲线（原始损失、对数尺度、损失比率）"""
    plt.figure(figsize=(15, 5))

    # 原始损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 对数尺度损失曲线
    plt.subplot(1, 3, 2)
    plt.semilogy(train_losses, label='Training Loss', linewidth=2)
    plt.semilogy(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Curves (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 损失比率
    plt.subplot(1, 3, 3)
    loss_ratio = [val/train if train > 0 else 0 for train, val in zip(train_losses, val_losses)]
    plt.plot(loss_ratio, linewidth=2, color='purple')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss / Train Loss')
    plt.title('Loss Ratio (Indication of Overfitting)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comprehensive_analysis(targets, predictions, train_losses, val_losses, base_save_path):
    """
    生成综合分析报告（包括残差图、误差分析、预测区间、学习曲线等）。
    返回包含各项指标（R², RMSE, MAE等）的字典。
    """
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
