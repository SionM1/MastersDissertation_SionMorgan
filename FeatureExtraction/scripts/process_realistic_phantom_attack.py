#!/usr/bin/env python3
"""
Process Realistic Phantom ECU Attack Dataset for Model Evaluation
Extracts features from the realistic attack and creates test files
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import feature_engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.feature_engineering import CANFeatureExtractor

def process_realistic_phantom_attack():
    """
    Process the realistic Phantom ECU attack dataset
    """
    print("Processing Realistic Phantom ECU Attack Dataset")
    print("=" * 60)
    
    # Initialize feature extractor
    extractor = CANFeatureExtractor(window_size=1.0)
    
    # Process realistic phantom attack
    realistic_attack_path = "../../AttackData/Realistic_Phantom_ECU_dataset.csv"
    
    if not os.path.exists(realistic_attack_path):
        print(f"Error: {realistic_attack_path} not found!")
        return None
    
    print(f"Processing realistic phantom attack data...")
    realistic_features = extractor.process_dataset(realistic_attack_path, 'realistic_phantom')
    
    if realistic_features is None:
        print("Failed to process realistic phantom attack!")
        return None
    
    # Save realistic phantom features
    output_path = "../data/features_realistic_phantom.csv"
    realistic_features.to_csv(output_path, index=False)
    print(f"Saved realistic phantom features: {output_path}")
    print(f"  Shape: {realistic_features.shape}")
    print(f"  Contamination rate: {(realistic_features['label'] != 'normal').mean()*100:.1f}%")
    
    return realistic_features

def create_test_dataset():
    """
    Create test dataset combining normal and realistic phantom attack data
    """
    print("\nCreating combined test dataset...")
    
    # Load normal features
    normal_path = "../data/features_normal.csv"
    if not os.path.exists(normal_path):
        print(f"Error: {normal_path} not found! Please run feature extraction first.")
        return
    
    normal_data = pd.read_csv(normal_path)
    print(f"Loaded normal data: {len(normal_data):,} samples")
    
    # Load realistic phantom features
    realistic_path = "../data/features_realistic_phantom.csv"
    if not os.path.exists(realistic_path):
        print(f"Error: {realistic_path} not found!")
        return
    
    realistic_data = pd.read_csv(realistic_path)
    print(f"Loaded realistic phantom data: {len(realistic_data):,} samples")
    
    # Sample data to create balanced test set
    # Use 20% of normal data and all realistic phantom data
    normal_sample_size = min(int(len(normal_data) * 0.2), 20000)
    normal_sample = normal_data.sample(n=normal_sample_size, random_state=42)
    
    # Combine datasets
    combined_data = pd.concat([normal_sample, realistic_data], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined test dataset
    test_output_path = "../data/test_normal_vs_realistic_phantom.csv"
    combined_data.to_csv(test_output_path, index=False)
    
    print(f"Created test dataset: {test_output_path}")
    print(f"  Total samples: {len(combined_data):,}")
    print(f"  Normal samples: {len(combined_data[combined_data['label'] == 'normal']):,}")
    print(f"  Attack samples: {len(combined_data[combined_data['label'] == 'realistic_phantom']):,}")
    print(f"  Contamination rate: {(combined_data['label'] != 'normal').mean()*100:.1f}%")

def main():
    """
    Main execution function
    """
    print("Realistic Phantom ECU Attack Feature Processing")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Process realistic phantom attack
    realistic_features = process_realistic_phantom_attack()
    
    if realistic_features is not None:
        # Create test dataset
        create_test_dataset()
        
        print("\nProcessing completed successfully!")
        print("Files created:")
        print("  - features_realistic_phantom.csv")
        print("  - test_normal_vs_realistic_phantom.csv")
    else:
        print("Processing failed!")

if __name__ == "__main__":
    main()
