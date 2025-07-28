#!/usr/bin/env python3
"""
Split Attack Data by Type and Extract Features
Creates separate feature files for each attack type: DoS, Fuzzy, Gear, RPM
"""

import pandas as pd
import numpy as np
import os
from feature_engineering import CANFeatureExtractor

def extract_features_for_attack_type(attack_type, input_file, output_file):
    """
    Extract features for a specific attack type
    """
    print(f"\nProcessing {attack_type} attack data...")
    
    # Initialize feature extractor
    extractor = CANFeatureExtractor(window_size=1.0)
    
    # Extract features using the existing method
    features_df = extractor.process_dataset(input_file, attack_type.lower())
    
    if features_df is not None:
        # Save features
        features_df.to_csv(output_file, index=False)
        
        print(f"   Extracted {len(features_df):,} feature records")
        print(f"   Saved to: {output_file}")
        
        return features_df
    else:
        print(f"   Failed to extract features for {attack_type}")
        return None

def create_attack_specific_datasets():
    """
    Create feature datasets for each attack type
    """
    print("Creating Attack-Specific Feature Datasets")
    print("=" * 50)
    
    # Define attack types and their corresponding files
    attack_types = {
        'DoS': '../AttackData/DoS_dataset.csv',
        'Fuzzy': '../AttackData/Fuzzy_dataset.csv', 
        'Gear': '../AttackData/gear_dataset.csv',
        'RPM': '../AttackData/RPM_dataset.csv'
    }
    
    attack_features = {}
    
    # Process each attack type
    for attack_type, input_file in attack_types.items():
        if os.path.exists(input_file):
            output_file = f'features_{attack_type.lower()}.csv'
            try:
                features_df = extract_features_for_attack_type(attack_type, input_file, output_file)
                attack_features[attack_type] = features_df
            except Exception as e:
                print(f"   Error processing {attack_type}: {e}")
        else:
            print(f"   Warning: {input_file} not found, skipping {attack_type}")
    
    # Create summary
    print(f"\n" + "=" * 50)
    print("Attack Data Summary:")
    for attack_type, features_df in attack_features.items():
        print(f"   {attack_type}: {len(features_df):,} feature records")
    
    return attack_features

def create_combined_test_datasets(attack_features, normal_features):
    """
    Create combined datasets for testing: normal + each attack type
    """
    print(f"\nCreating Combined Test Datasets...")
    
    for attack_type, attack_df in attack_features.items():
        # Combine normal and specific attack type
        combined_df = pd.concat([normal_features, attack_df], ignore_index=True)
        
        # Shuffle the data
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        output_file = f'features_normal_vs_{attack_type.lower()}.csv'
        combined_df.to_csv(output_file, index=False)
        
        normal_count = len(normal_features)
        attack_count = len(attack_df)
        contamination = attack_count / (normal_count + attack_count) * 100
        
        print(f"   {attack_type}: {len(combined_df):,} total ({normal_count:,} normal + {attack_count:,} attack, {contamination:.1f}% contamination)")
        print(f"   Saved to: {output_file}")

def main():
    """
    Main execution function
    """
    # Create attack-specific feature datasets
    attack_features = create_attack_specific_datasets()
    
    if not attack_features:
        print("No attack features were extracted. Please check input files.")
        return
    
    # Load normal features
    normal_file = 'features_normal.csv'
    if os.path.exists(normal_file):
        print(f"\nLoading normal features from {normal_file}...")
        normal_features = pd.read_csv(normal_file)
        print(f"   Loaded {len(normal_features):,} normal feature records")
        
        # Create combined test datasets
        create_combined_test_datasets(attack_features, normal_features)
        
    else:
        print(f"Warning: {normal_file} not found. Cannot create combined datasets.")
    
    print(f"\nAttack data splitting completed!")
    print("Created files:")
    for attack_type in attack_features.keys():
        print(f"   - features_{attack_type.lower()}.csv")
        print(f"   - features_normal_vs_{attack_type.lower()}.csv")

if __name__ == "__main__":
    main()
