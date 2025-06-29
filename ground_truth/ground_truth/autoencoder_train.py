#!/usr/bin/env python3
"""
Lidar Autoencoder Compression Evaluation Script

This script reads lidar data from CSV, truncates last 2 values, and evaluates 
autoencoder compression at 16, 32, and 64 dimensions with detailed plots.

Usage: python lidar_compression_eval.py <csv_file_path>
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import argparse

class LidarAutoencoder(nn.Module):
    def __init__(self, input_dim=1078, encoding_dim=64):
        super(LidarAutoencoder, self).__init__()
        
        # Calculate intermediate dimensions based on encoding size
        dim1 = max(512, encoding_dim * 8)
        dim2 = max(256, encoding_dim * 4)
        dim3 = max(128, encoding_dim * 2)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            nn.Linear(dim3, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, dim3),
            nn.ReLU(),
            nn.Linear(dim3, dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim1, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            return encoded.detach()

class LidarCompressor:
    def __init__(self, encoding_dim=64, input_dim=1078):
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.model = LidarAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.training_losses = []
        
    def prepare_data(self, lidar_data):
        normalized_data = self.scaler.fit_transform(lidar_data)
        tensor_data = torch.FloatTensor(normalized_data)
        return tensor_data
    
    def train(self, lidar_data, epochs=50, batch_size=32, learning_rate=0.001, verbose=True):
        tensor_data = self.prepare_data(lidar_data)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.training_losses = []
        self.model.train()
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                print(f'Dim {self.encoding_dim} - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Time: {elapsed:.1f}s')
        
        self.model.eval()
        training_time = time.time() - start_time
        return self.training_losses, training_time
    
    def compress(self, lidar_data):
        normalized_data = self.scaler.transform(lidar_data)
        tensor_data = torch.FloatTensor(normalized_data).to(self.device)
        with torch.no_grad():
            encoded = self.model.encode(tensor_data)
        return encoded.cpu().detach().numpy()
    
    def decompress(self, encoded_data):
        tensor_encoded = torch.FloatTensor(encoded_data).to(self.device)
        with torch.no_grad():
            decoded = self.model.decoder(tensor_encoded)
        decoded_np = decoded.cpu().detach().numpy()
        reconstructed = self.scaler.inverse_transform(decoded_np)
        return reconstructed
    
    def evaluate_compression(self, test_data):
        encoded = self.compress(test_data)
        reconstructed = self.decompress(encoded)
        
        # Calculate metrics
        mse = np.mean((test_data - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_data - reconstructed))
        
        # R-squared
        ss_res = np.sum((test_data - reconstructed) ** 2)
        ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Compression metrics
        original_size = test_data.nbytes
        compressed_size = encoded.nbytes
        compression_ratio = original_size / compressed_size
        
        return {
            'compression_ratio': compression_ratio,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'encoded': encoded,
            'reconstructed': reconstructed,
            'original': test_data
        }
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'encoding_dim': self.encoding_dim,
            'input_dim': self.input_dim
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model and scaler"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.encoding_dim = checkpoint['encoding_dim']
        self.input_dim = checkpoint['input_dim']

def load_and_preprocess_data(csv_path):
    """Load CSV data and truncate last 2 values from each row"""
    print(f"Loading data from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    
    # Convert to numpy array and truncate last 2 columns
    data = df.values
    if data.shape[1] >= 2:
        data = data[:, :-2]  # Remove last 2 columns
        print(f"After truncation: {data.shape}")
    else:
        raise ValueError("Data must have at least 2 columns to truncate")
    
    return data

def evaluate_compressions(data, encoding_dims=[16, 32, 64], test_size=0.2):
    """Evaluate different compression dimensions"""
    print(f"\nEvaluating compressions for dimensions: {encoding_dims}")
    print(f"Total samples: {len(data)}")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    results = {}
    
    for dim in encoding_dims:
        print(f"\n{'='*50}")
        print(f"Training autoencoder with {dim} dimensions...")
        print(f"{'='*50}")
        
        # Create and train compressor
        compressor = LidarCompressor(encoding_dim=dim, input_dim=data.shape[1])
        losses, training_time = compressor.train(train_data, epochs=50, verbose=True)
        
        # Evaluate on test data
        eval_results = compressor.evaluate_compression(test_data)
        eval_results['training_losses'] = losses
        eval_results['training_time'] = training_time
        eval_results['compressor'] = compressor
        
        results[dim] = eval_results
        
        # Save the trained model
        model_filename = f'lidar_autoencoder_{dim}D.pth'
        compressor.save_model(model_filename)
        eval_results['model_filename'] = model_filename
        
        print(f"\nResults for {dim}D compression:")
        print(f"Compression Ratio: {eval_results['compression_ratio']:.2f}x")
        print(f"RMSE: {eval_results['rmse']:.6f}")
        print(f"MAE: {eval_results['mae']:.6f}")
        print(f"R²: {eval_results['r_squared']:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Model saved to: {model_filename}")
    
    return results, train_data, test_data

def create_comprehensive_plots(results, encoding_dims, test_data, output_prefix='compression'):
    """Create separate plots for each visualization"""
    plots_created = []
    
    # 1. Training Loss Comparison
    plt.figure(figsize=(10, 6))
    for dim in encoding_dims:
        plt.plot(results[dim]['training_losses'], label=f'{dim}D', linewidth=2)
    plt.title('Training Loss Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_training_loss.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 2. Compression Ratio vs Quality
    plt.figure(figsize=(8, 6))
    ratios = [results[dim]['compression_ratio'] for dim in encoding_dims]
    r_squared = [results[dim]['r_squared'] for dim in encoding_dims]
    
    plt.scatter(ratios, r_squared, s=100, c=['red', 'orange', 'green'], alpha=0.7)
    for i, dim in enumerate(encoding_dims):
        plt.annotate(f'{dim}D', (ratios[i], r_squared[i]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    plt.title('Compression vs Quality Trade-off', fontsize=14, fontweight='bold')
    plt.xlabel('Compression Ratio')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_compression_vs_quality.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 3. Error Metrics Comparison
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(encoding_dims))
    width = 0.35
    
    rmse_values = [results[dim]['rmse'] for dim in encoding_dims]
    mae_values = [results[dim]['mae'] for dim in encoding_dims]
    
    plt.bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    plt.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.8)
    
    plt.title('Error Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Encoding Dimensions')
    plt.ylabel('Error Value')
    plt.xticks(x_pos, [f'{dim}D' for dim in encoding_dims])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_error_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 4. Compression Ratios
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(encoding_dims)), ratios, color=['red', 'orange', 'green'], alpha=0.7)
    plt.title('Compression Ratios', fontsize=14, fontweight='bold')
    plt.xlabel('Encoding Dimensions')
    plt.ylabel('Compression Ratio')
    plt.xticks(range(len(encoding_dims)), [f'{dim}D' for dim in encoding_dims])
    for i, ratio in enumerate(ratios):
        plt.text(i, ratio + max(ratios)*0.02, f'{ratio:.1f}x', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_compression_ratios.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 5-7. Reconstruction Error Heatmaps for each dimension
    for i, dim in enumerate(encoding_dims):
        plt.figure(figsize=(12, 8))
        original = test_data[:100]  # First 100 samples
        reconstructed = results[dim]['reconstructed'][:100]
        
        # Calculate point-wise absolute error for each sample
        errors = np.abs(original - reconstructed)
        
        # Plot error heatmap
        im = plt.imshow(errors.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im)
        plt.title(f'{dim}D: Reconstruction Error Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Lidar Point Index')
        plt.tight_layout()
        filename = f'{output_prefix}_error_heatmap_{dim}D.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plots_created.append(filename)
        plt.show()
        plt.close()
        
    # 8. Original vs Reconstructed for best model
    best_dim = max(encoding_dims, key=lambda d: results[d]['r_squared'])
    plt.figure(figsize=(12, 6))
    sample_idx = 0
    original = test_data[sample_idx]
    reconstructed = results[best_dim]['reconstructed'][sample_idx]
    
    plt.plot(original, label='Original', alpha=0.8, linewidth=2)
    plt.plot(reconstructed, label='Reconstructed', alpha=0.8, linewidth=2)
    plt.title(f'{best_dim}D: Original vs Reconstructed (Best Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Lidar Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_original_vs_reconstructed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 9. Mean Absolute Error per Lidar Point
    plt.figure(figsize=(12, 6))
    for dim in encoding_dims:
        original = test_data
        reconstructed = results[dim]['reconstructed']
        point_wise_mae = np.mean(np.abs(original - reconstructed), axis=0)
        plt.plot(point_wise_mae, label=f'{dim}D', alpha=0.8, linewidth=2)
    
    plt.title('Mean Absolute Error per Lidar Point', fontsize=14, fontweight='bold')
    plt.xlabel('Lidar Point Index')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_mae_per_point.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 10. Error Distribution Comparison
    plt.figure(figsize=(10, 6))
    for dim in encoding_dims:
        errors = test_data - results[dim]['reconstructed']
        plt.hist(errors.flatten(), bins=50, alpha=0.6, label=f'{dim}D', density=True)
    plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_error_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 11. Sample-wise RMSE comparison
    plt.figure(figsize=(12, 6))
    for dim in encoding_dims:
        sample_rmse = np.sqrt(np.mean((test_data - results[dim]['reconstructed'])**2, axis=1))
        plt.plot(sample_rmse[:150], label=f'{dim}D', alpha=0.8)
    plt.title('Per-Sample RMSE (First 150 samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_sample_rmse.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 12. Max Error per Sample
    plt.figure(figsize=(12, 6))
    for dim in encoding_dims:
        max_errors = np.max(np.abs(test_data - results[dim]['reconstructed']), axis=1)
        plt.plot(max_errors[:150], label=f'{dim}D', alpha=0.8, marker='o', markersize=3)
    plt.title('Maximum Error per Sample', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Maximum Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_max_error.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 13. Training Time Comparison
    plt.figure(figsize=(8, 6))
    times = [results[dim]['training_time'] for dim in encoding_dims]
    plt.bar(range(len(encoding_dims)), times, color=['red', 'orange', 'green'], alpha=0.7)
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Encoding Dimensions')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(range(len(encoding_dims)), [f'{dim}D' for dim in encoding_dims])
    for i, time_val in enumerate(times):
        plt.text(i, time_val + max(times)*0.02, f'{time_val:.1f}s', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_training_time.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 14. Latent Space Visualization (first 2 dimensions)
    plt.figure(figsize=(8, 6))
    best_encoded = results[best_dim]['encoded']
    scatter = plt.scatter(best_encoded[:200, 0], best_encoded[:200, 1], 
                         alpha=0.6, s=20, c=range(200), cmap='viridis')
    plt.title(f'Latent Space Visualization ({best_dim}D)', fontsize=14, fontweight='bold')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_latent_space.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 15. Reconstruction Quality Metrics
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(encoding_dims))
    width = 0.25
    
    r2_values = [results[dim]['r_squared'] for dim in encoding_dims]
    rmse_values = [results[dim]['rmse'] / max([results[d]['rmse'] for d in encoding_dims]) for dim in encoding_dims]  # Normalized
    mae_values = [results[dim]['mae'] / max([results[d]['mae'] for d in encoding_dims]) for dim in encoding_dims]    # Normalized
    
    plt.bar(x_pos - width, r2_values, width, label='R² Score', alpha=0.8)
    plt.bar(x_pos, rmse_values, width, label='RMSE (normalized)', alpha=0.8)
    plt.bar(x_pos + width, mae_values, width, label='MAE (normalized)', alpha=0.8)
    
    plt.title('Normalized Quality Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Encoding Dimensions')
    plt.ylabel('Metric Value')
    plt.xticks(x_pos, [f'{dim}D' for dim in encoding_dims])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'{output_prefix}_quality_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # 16. Original vs Reconstructed for each dimension (comparison)
    fig, axes = plt.subplots(len(encoding_dims), 1, figsize=(12, 4*len(encoding_dims)))
    if len(encoding_dims) == 1:
        axes = [axes]
    
    sample_idx = 0
    for i, dim in enumerate(encoding_dims):
        original = test_data[sample_idx]
        reconstructed = results[dim]['reconstructed'][sample_idx]
        
        axes[i].plot(original, label='Original', alpha=0.8, linewidth=2)
        axes[i].plot(reconstructed, label='Reconstructed', alpha=0.8, linewidth=2)
        axes[i].set_title(f'{dim}D: Original vs Reconstructed', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Lidar Point Index')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_prefix}_all_comparisons.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
def create_compression_demonstration(results, encoding_dims, test_data, output_prefix='compression'):
    """Create compressed versions and demonstrate compression/decompression cycle"""
    print(f"\n{'='*60}")
    print("COMPRESSION/DECOMPRESSION DEMONSTRATION")
    print(f"{'='*60}")
    
    compressed_data = {}
    plots_created = []
    
    for dim in encoding_dims:
        print(f"\nProcessing {dim}D compression...")
        
        # Get the trained compressor
        compressor = results[dim]['compressor']
        
        # Compress the entire test dataset
        compressed = compressor.compress(test_data)
        decompressed = compressor.decompress(compressed)
        
        # Store compressed data
        compressed_data[dim] = {
            'compressed': compressed,
            'decompressed': decompressed,
            'original': test_data
        }
        
        # Calculate compression statistics
        original_size = test_data.nbytes
        compressed_size = compressed.nbytes
        compression_ratio = original_size / compressed_size
        
        print(f"  Original size: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
        print(f"  Compressed size: {compressed_size:,} bytes ({compressed_size/1024/1024:.2f} MB)")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print(f"  Space saved: {(1 - compressed_size/original_size)*100:.1f}%")
        
        # Create detailed comparison plots for this dimension
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{dim}D Compression: Original vs Decompressed Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Sample comparison (first sample)
        sample_idx = 0
        axes[0, 0].plot(test_data[sample_idx], label='Original', linewidth=2, alpha=0.8)
        axes[0, 0].plot(decompressed[sample_idx], label='Decompressed', linewidth=2, alpha=0.8, linestyle='--')
        axes[0, 0].set_title(f'Sample {sample_idx}: Original vs Decompressed')
        axes[0, 0].set_xlabel('Lidar Point Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error for this sample
        error = test_data[sample_idx] - decompressed[sample_idx]
        axes[0, 1].plot(error, color='red', linewidth=2)
        axes[0, 1].set_title(f'Reconstruction Error (Sample {sample_idx})')
        axes[0, 1].set_xlabel('Lidar Point Index')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Multiple samples overlay
        num_samples_to_show = min(5, len(test_data))
        for i in range(num_samples_to_show):
            alpha_val = 0.7 - i * 0.1
            axes[1, 0].plot(test_data[i], alpha=alpha_val, linewidth=1, color='blue', label='Original' if i == 0 else "")
            axes[1, 0].plot(decompressed[i], alpha=alpha_val, linewidth=1, color='red', linestyle='--', label='Decompressed' if i == 0 else "")
        axes[1, 0].set_title(f'Multiple Samples Overlay (First {num_samples_to_show})')
        axes[1, 0].set_xlabel('Lidar Point Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error statistics
        all_errors = test_data - decompressed
        mean_error = np.mean(all_errors, axis=0)
        std_error = np.std(all_errors, axis=0)
        
        axes[1, 1].plot(mean_error, label='Mean Error', linewidth=2, color='red')
        axes[1, 1].fill_between(range(len(mean_error)), 
                               mean_error - std_error, 
                               mean_error + std_error, 
                               alpha=0.3, color='red', label='±1 Std')
        axes[1, 1].set_title('Error Statistics Across All Samples')
        axes[1, 1].set_xlabel('Lidar Point Index')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        filename = f'{output_prefix}_detailed_comparison_{dim}D.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plots_created.append(filename)
        plt.show()
        plt.close()
        
        # Create error distribution plot for this dimension
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(all_errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{dim}D: Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        sample_mse = np.mean(all_errors**2, axis=1)
        plt.plot(sample_mse[:100], marker='o', markersize=3)
        plt.title(f'{dim}D: MSE per Sample (First 100)')
        plt.xlabel('Sample Index')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        point_mse = np.mean(all_errors**2, axis=0)
        plt.plot(point_mse)
        plt.title(f'{dim}D: MSE per Lidar Point')
        plt.xlabel('Lidar Point Index')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{output_prefix}_error_analysis_{dim}D.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plots_created.append(filename)
        plt.show()
        plt.close()
    
    # Create overall comparison plot
    print(f"\nCreating overall comparison plots...")
    
    # Plot comparing all dimensions side by side
    fig, axes = plt.subplots(len(encoding_dims), 3, figsize=(18, 5*len(encoding_dims)))
    if len(encoding_dims) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Compression Comparison Across All Dimensions', fontsize=16, fontweight='bold')
    
    sample_idx = 0
    for i, dim in enumerate(encoding_dims):
        original = test_data[sample_idx]
        decompressed = compressed_data[dim]['decompressed'][sample_idx]
        error = original - decompressed
        
        # Original vs Decompressed
        axes[i, 0].plot(original, label='Original', linewidth=2)
        axes[i, 0].plot(decompressed, label='Decompressed', linewidth=2, linestyle='--')
        axes[i, 0].set_title(f'{dim}D: Original vs Decompressed')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Error
        axes[i, 1].plot(error, color='red', linewidth=2)
        axes[i, 1].set_title(f'{dim}D: Reconstruction Error')
        axes[i, 1].set_ylabel('Error')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Compressed representation visualization (first 32 dimensions)
        compressed_sample = compressed_data[dim]['compressed'][sample_idx]
        display_dims = min(32, len(compressed_sample))
        axes[i, 2].bar(range(display_dims), compressed_sample[:display_dims], alpha=0.7)
        axes[i, 2].set_title(f'{dim}D: Compressed Representation (First {display_dims} dims)')
        axes[i, 2].set_ylabel('Compressed Value')
        axes[i, 2].grid(True, alpha=0.3)
        
        if i == len(encoding_dims) - 1:
            axes[i, 0].set_xlabel('Lidar Point Index')
            axes[i, 1].set_xlabel('Lidar Point Index')
            axes[i, 2].set_xlabel('Compressed Dimension')
    
    plt.tight_layout()
    filename = f'{output_prefix}_all_dimensions_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plots_created.append(filename)
    plt.show()
    plt.close()
    
    # Save compressed datasets
    print(f"\nSaving compressed datasets...")
    for dim in encoding_dims:
        # Save compressed data
        compressed_filename = f'{output_prefix}_compressed_data_{dim}D.npy'
        np.save(compressed_filename, compressed_data[dim]['compressed'])
        print(f"  Compressed data ({dim}D): {compressed_filename}")
        
        # Save decompressed data for verification
        decompressed_filename = f'{output_prefix}_decompressed_data_{dim}D.npy'
        np.save(decompressed_filename, compressed_data[dim]['decompressed'])
        print(f"  Decompressed data ({dim}D): {decompressed_filename}")
    
    # Create summary statistics
    print(f"\n{'='*40}")
    print("COMPRESSION SUMMARY")
    print(f"{'='*40}")
    
    summary_data = []
    for dim in encoding_dims:
        original = test_data
        decompressed = compressed_data[dim]['decompressed']
        
        mse = np.mean((original - decompressed)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original - decompressed))
        max_error = np.max(np.abs(original - decompressed))
        
        compression_ratio = original.nbytes / compressed_data[dim]['compressed'].nbytes
        
        summary_data.append({
            'Dimension': f'{dim}D',
            'Compression Ratio': f'{compression_ratio:.1f}x',
            'RMSE': f'{rmse:.6f}',
            'MAE': f'{mae:.6f}',
            'Max Error': f'{max_error:.6f}',
            'R²': f'{results[dim]["r_squared"]:.4f}'
        })
        
        print(f"{dim}D: {compression_ratio:.1f}x compression, RMSE={rmse:.6f}, MAE={mae:.6f}")
    
    return compressed_data, plots_created, summary_data

def main():
    parser = argparse.ArgumentParser(description='Evaluate lidar autoencoder compression')
    parser.add_argument('csv_file', help='Path to CSV file containing lidar data')
    parser.add_argument('--dims', nargs='+', type=int, default=[16, 32, 64],
                       help='Compression dimensions to evaluate (default: 16 32 64)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--output', type=str, default='compression_results.png',
                       help='Output filename for plots (default: compression_results.png)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found!")
        sys.exit(1)
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data(args.csv_file)
        
        # Evaluate compressionsfcs
        results, train_data, test_data = evaluate_compressions(data, args.dims)
        
        # Create plots
        print(f"\nGenerating separate plots...")
        plots_created = create_comprehensive_plots(results, args.dims, test_data, 
                                                  output_prefix=args.output.replace('.png', ''))
        
        # Create compression demonstration  
        compressed_data, demo_plots, summary_data = create_compression_demonstration(
            results, args.dims, test_data, output_prefix=args.output.replace('.png', ''))
        
        # Combine all plot lists
        if plots_created is None:
            plots_created = []
        if demo_plots is None:
            demo_plots = []
            
        all_plots = plots_created + demo_plots
        
        print(f"\nAll plots created:")
        for plot_file in all_plots:
            print(f"  - {plot_file}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        best_quality = max(args.dims, key=lambda d: results[d]['r_squared'])
        best_compression = max(args.dims, key=lambda d: results[d]['compression_ratio'])
        
        print(f"Best Quality: {best_quality}D (R² = {results[best_quality]['r_squared']:.4f})")
        print(f"Best Compression: {best_compression}D ({results[best_compression]['compression_ratio']:.1f}x)")
        print(f"Recommended: 32D for balanced compression and quality")
        
        print(f"\nTrained models saved:")
        for dim in args.dims:
            print(f"  - {results[dim]['model_filename']}")
        
        print(f"\nTo use a saved model:")
        print(f"compressor = LidarCompressor(encoding_dim={best_quality})")
        print(f"compressor.load_model('{results[best_quality]['model_filename']}')")
        print(f"compressed = compressor.compress(your_data)")
        print(f"reconstructed = compressor.decompress(compressed)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()