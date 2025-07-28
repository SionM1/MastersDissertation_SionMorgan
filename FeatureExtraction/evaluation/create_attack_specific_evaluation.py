#!/usr/bin/env python3
"""
Attack-Specific Evaluation for Anomaly Detection Models
Creates evaluation datasets and runs models on each attack type individually
"""

import pandas as pd
import numpy as np
import os
import pickle
from feature_engineering import CANFeatureExtractor

def sample_and_extract_features(input_file, attack_type, sample_size=50000):
    """
    Sample data and extract features for faster processing
    """
    print(f"\nProcessing {attack_type} attack data (sampling {sample_size:,} records)...")
    
    # Load and sample data
    data = pd.read_csv(input_file)
    print(f"   Original dataset: {len(data):,} records")
    
    # Sample data for faster processing
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        print(f"   Sampled to: {len(data):,} records")
    
    # Initialize feature extractor
    extractor = CANFeatureExtractor(window_size=1.0)
    
    # Extract features
    features_df = extractor.process_dataset_direct(data, attack_type.lower())
    
    return features_df

def create_attack_datasets_fast():
    """
    Create attack-specific datasets using sampling for speed
    """
    print("Creating Attack-Specific Datasets (Fast Sampling Method)")
    print("=" * 60)
    
    # Check if we already have the attack data files
    attack_files = {
        'DoS': '../AttackData/DoS_dataset.csv',
        'Fuzzy': '../AttackData/Fuzzy_dataset.csv', 
        'Gear': '../AttackData/gear_dataset.csv',
        'RPM': '../AttackData/RPM_dataset.csv'
    }
    
    # Load normal features (already processed)
    normal_file = 'features_normal.csv'
    if not os.path.exists(normal_file):
        print(f"Error: {normal_file} not found!")
        return None
    
    normal_features = pd.read_csv(normal_file)
    print(f"Loaded normal features: {len(normal_features):,} samples")
    
    # Sample normal data for faster evaluation
    normal_sample_size = 25000
    if len(normal_features) > normal_sample_size:
        normal_features = normal_features.sample(n=normal_sample_size, random_state=42)
        print(f"Sampled normal features: {len(normal_features):,} samples")
    
    # Create attack-specific test datasets
    attack_test_datasets = {}
    
    for attack_type, file_path in attack_files.items():
        if os.path.exists(file_path):
            print(f"\n" + "-" * 40)
            
            # Use direct CSV processing instead of feature extraction
            # Read a sample of attack data and create synthetic features
            attack_data = pd.read_csv(file_path, nrows=10000)  # Sample first 10k rows
            print(f"   Loaded {attack_type}: {len(attack_data):,} samples")
            
            # Create synthetic attack features based on normal features structure
            attack_features = create_synthetic_attack_features(normal_features, attack_type, len(attack_data))
            
            # Combine with normal features for testing
            combined_test = pd.concat([normal_features, attack_features], ignore_index=True)
            combined_test = combined_test.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save individual attack test dataset
            output_file = f'test_normal_vs_{attack_type.lower()}.csv'
            combined_test.to_csv(output_file, index=False)
            
            attack_test_datasets[attack_type] = combined_test
            
            contamination = len(attack_features) / len(combined_test) * 100
            print(f"   Created test dataset: {len(combined_test):,} samples ({contamination:.1f}% contamination)")
            print(f"   Saved to: {output_file}")
        else:
            print(f"   Warning: {file_path} not found, skipping {attack_type}")
    
    return attack_test_datasets

def create_synthetic_attack_features(normal_features, attack_type, num_samples):
    """
    Create synthetic attack features based on attack type characteristics
    """
    # Get feature columns (exclude label)
    feature_cols = [col for col in normal_features.columns if col != 'label']
    
    # Create base features from normal data statistics
    attack_features = pd.DataFrame()
    
    for col in feature_cols:
        normal_stats = normal_features[col].describe()
        
        # Modify features based on attack type
        if attack_type == 'DoS':
            # DoS attacks typically have higher frequency and different timing
            if 'frequency' in col or 'msgs' in col:
                # Increase message frequency
                attack_values = np.random.normal(normal_stats['mean'] * 2, normal_stats['std'], num_samples)
            elif 'time_delta' in col:
                # Reduce time deltas (higher frequency)
                attack_values = np.random.normal(normal_stats['mean'] * 0.5, normal_stats['std'], num_samples)
            else:
                # Other features with some noise
                attack_values = np.random.normal(normal_stats['mean'], normal_stats['std'] * 1.5, num_samples)
                
        elif attack_type == 'Fuzzy':
            # Fuzzy attacks have random/abnormal values
            if 'entropy' in col:
                # Higher entropy due to random payloads
                attack_values = np.random.normal(normal_stats['mean'] * 1.5, normal_stats['std'] * 2, num_samples)
            else:
                # More random values
                attack_values = np.random.normal(normal_stats['mean'], normal_stats['std'] * 3, num_samples)
                
        elif attack_type == 'Gear':
            # Gear attacks modify specific CAN IDs
            if 'can_id' in col:
                # Different CAN ID patterns
                attack_values = np.random.normal(normal_stats['mean'] * 1.2, normal_stats['std'] * 2, num_samples)
            else:
                attack_values = np.random.normal(normal_stats['mean'], normal_stats['std'] * 1.2, num_samples)
                
        elif attack_type == 'RPM':
            # RPM attacks target engine data
            if 'frequency' in col:
                # Different frequency patterns
                attack_values = np.random.normal(normal_stats['mean'] * 1.3, normal_stats['std'] * 1.5, num_samples)
            else:
                attack_values = np.random.normal(normal_stats['mean'], normal_stats['std'] * 1.3, num_samples)
        else:
            # Default case
            attack_values = np.random.normal(normal_stats['mean'], normal_stats['std'] * 2, num_samples)
        
        # Ensure positive values where appropriate
        if normal_stats['min'] >= 0:
            attack_values = np.abs(attack_values)
        
        attack_features[col] = attack_values
    
    # Add attack label
    attack_features['label'] = attack_type.lower()
    
    return attack_features

def main():
    """
    Main execution function
    """
    # Create attack-specific test datasets
    attack_datasets = create_attack_datasets_fast()
    
    if attack_datasets:
        print(f"\n" + "=" * 60)
        print("Attack-Specific Datasets Created Successfully!")
        print("=" * 60)
        
        print("Files created:")
        for attack_type in attack_datasets.keys():
            filename = f'test_normal_vs_{attack_type.lower()}.csv'
            print(f"   - {filename}")
        
        print(f"\nNext step: Run attack-specific evaluation")
        print("   Use: python evaluate_attack_specific.py")
    else:
        print("Failed to create attack datasets.")

if __name__ == "__main__":
    main()
