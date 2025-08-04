#!/usr/bin/env python3
"""
Normal Traffic Extractor
Extracts clean normal CAN bus traffic from car hacking attack datasets
to create a realistic baseline for generating sophisticated attacks.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class NormalTrafficExtractor:
    def __init__(self):
        """Initialize the normal traffic extractor"""
        self.attack_datasets = [
            'DoS_dataset.csv',
            'Fuzzy_dataset.csv', 
            'gear_dataset.csv',
            'RPM_dataset.csv'
        ]
        self.feature_datasets = [
            '../FeatureExtraction/data/test_normal_vs_dos.csv',
            '../FeatureExtraction/data/test_normal_vs_fuzzy.csv',
            '../FeatureExtraction/data/test_normal_vs_gear.csv'
        ]
        
    def extract_from_labeled_features(self):
        """
        Extract normal traffic data from labeled feature datasets
        These contain 'normal' vs attack labels for easy extraction
        """
        print("Extracting normal traffic from labeled feature datasets...")
        
        all_normal_features = []
        
        for dataset_path in self.feature_datasets:
            if os.path.exists(dataset_path):
                print(f"Processing: {dataset_path}")
                
                try:
                    df = pd.read_csv(dataset_path)
                    
                    # Extract only normal labeled data
                    normal_data = df[df['label'] == 'normal'].copy()
                    normal_data['source_dataset'] = os.path.basename(dataset_path)
                    
                    all_normal_features.append(normal_data)
                    print(f"  - Found {len(normal_data)} normal samples")
                    
                except Exception as e:
                    print(f"  - Error processing {dataset_path}: {e}")
            else:
                print(f"  - File not found: {dataset_path}")
        
        if all_normal_features:
            combined_normal = pd.concat(all_normal_features, ignore_index=True)
            
            # Remove duplicate samples to get clean normal data
            before_dedup = len(combined_normal)
            combined_normal = combined_normal.drop_duplicates(
                subset=['time_delta', 'msg_frequency', 'unique_ids_in_window', 
                       'payload_entropy', 'can_id_std']
            )
            after_dedup = len(combined_normal)
            
            print(f"Combined normal samples: {before_dedup}")
            print(f"After deduplication: {after_dedup}")
            
            # Save extracted normal features
            output_path = 'extracted_normal_features.csv'
            combined_normal.to_csv(output_path, index=False)
            print(f"Saved normal features to: {output_path}")
            
            return combined_normal
        else:
            print("No normal feature data found!")
            return None
    
    def analyze_normal_patterns(self, normal_data):
        """
        Analyze normal CAN traffic patterns to understand baseline behavior
        """
        if normal_data is None:
            return None
            
        print("\n" + "="*60)
        print("NORMAL CAN TRAFFIC PATTERN ANALYSIS")
        print("="*60)
        
        # Statistical analysis of normal traffic characteristics
        analysis = {
            'total_samples': len(normal_data),
            'time_delta': {
                'mean': normal_data['time_delta'].mean(),
                'std': normal_data['time_delta'].std(),
                'min': normal_data['time_delta'].min(),
                'max': normal_data['time_delta'].max(),
                'median': normal_data['time_delta'].median()
            },
            'msg_frequency': {
                'mean': normal_data['msg_frequency'].mean(),
                'std': normal_data['msg_frequency'].std(),
                'min': normal_data['msg_frequency'].min(),
                'max': normal_data['msg_frequency'].max(),
                'unique_values': sorted(normal_data['msg_frequency'].unique())
            },
            'unique_ids_in_window': {
                'mean': normal_data['unique_ids_in_window'].mean(),
                'std': normal_data['unique_ids_in_window'].std(),
                'min': normal_data['unique_ids_in_window'].min(),
                'max': normal_data['unique_ids_in_window'].max(),
                'most_common': normal_data['unique_ids_in_window'].mode().iloc[0] if len(normal_data) > 0 else None
            },
            'payload_entropy': {
                'mean': normal_data['payload_entropy'].mean(),
                'std': normal_data['payload_entropy'].std(),
                'min': normal_data['payload_entropy'].min(),
                'max': normal_data['payload_entropy'].max(),
                'median': normal_data['payload_entropy'].median()
            },
            'can_id_std': {
                'mean': normal_data['can_id_std'].mean(),
                'std': normal_data['can_id_std'].std(),
                'min': normal_data['can_id_std'].min(),
                'max': normal_data['can_id_std'].max()
            }
        }
        
        # Print detailed analysis
        for feature, stats in analysis.items():
            if feature == 'total_samples':
                print(f"Total Normal Samples: {stats:,}")
                continue
                
            print(f"\n{feature.upper().replace('_', ' ')}:")
            if isinstance(stats, dict):
                for stat_name, value in stats.items():
                    if stat_name == 'unique_values' and len(value) > 10:
                        print(f"  {stat_name}: {value[:10]}... (showing first 10 of {len(value)})")
                    elif isinstance(value, float):
                        print(f"  {stat_name}: {value:.6f}")
                    else:
                        print(f"  {stat_name}: {value}")
        
        return analysis
    
    def synthesize_normal_raw_traffic(self, normal_features, num_messages=100000):
        """
        Synthesize realistic raw CAN traffic from normal feature patterns
        This creates the baseline for our Phantom ECU attack generation
        """
        print(f"\nSynthesizing {num_messages:,} normal CAN messages from patterns...")
        
        if normal_features is None or len(normal_features) == 0:
            print("Error: No normal features available for synthesis")
            return None
        
        # Analyze patterns for synthesis
        analysis = self.analyze_normal_patterns(normal_features)
        
        # Common CAN IDs found in real automotive networks
        common_can_ids = [
            '0002', '0050', '0130', '0131', '0140', '0153', '018F', '01A0',
            '0260', '0280', '02A0', '02C0', '0316', '0320', '0329', '0350',
            '0370', '03D0', '0420', '043F', '0440', '0470', '0545', '05A0'
        ]
        
        # Generate synthetic normal traffic
        synthetic_messages = []
        base_timestamp = 1478198376.0  # Start from realistic timestamp
        
        for i in range(num_messages):
            # Sample timing characteristics from normal patterns
            time_delta_sample = np.random.normal(
                analysis['time_delta']['mean'], 
                analysis['time_delta']['std']
            )
            time_delta = max(0.0001, min(0.01, time_delta_sample))  # Clamp to realistic range
            
            # Calculate timestamp
            timestamp = base_timestamp + (i * time_delta)
            
            # Select CAN ID based on realistic frequency distribution
            can_id = np.random.choice(common_can_ids, p=self._get_can_id_probabilities())
            
            # Generate realistic payload based on normal entropy patterns
            payload = self._generate_normal_payload(analysis['payload_entropy'])
            
            synthetic_messages.append({
                'Timestamp': timestamp,
                'CAN_ID': can_id,
                'DLC': 8,  # Standard CAN frame length
                'Payload': payload
            })
        
        # Convert to DataFrame
        synthetic_df = pd.DataFrame(synthetic_messages)
        
        # Save synthetic normal traffic
        output_path = 'synthesized_normal_traffic.csv'
        synthetic_df.to_csv(output_path, index=False)
        
        print(f"Synthesized normal traffic saved to: {output_path}")
        print(f"Message count: {len(synthetic_df):,}")
        print(f"Time span: {synthetic_df['Timestamp'].max() - synthetic_df['Timestamp'].min():.2f} seconds")
        print(f"Unique CAN IDs: {synthetic_df['CAN_ID'].nunique()}")
        
        return synthetic_df
    
    def _get_can_id_probabilities(self):
        """Get realistic CAN ID frequency distribution"""
        # Based on real automotive CAN traffic patterns
        # More frequent IDs get higher probability
        probs = [
            0.15, 0.12, 0.08, 0.08, 0.06, 0.05, 0.05, 0.04,  # High frequency
            0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02,  # Medium frequency  
            0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01   # Low frequency
        ]
        return np.array(probs) / sum(probs)  # Normalize
    
    def _generate_normal_payload(self, entropy_stats):
        """Generate realistic CAN payload based on normal entropy patterns"""
        # Target entropy around normal mean
        target_entropy = np.random.normal(entropy_stats['mean'], entropy_stats['std'] * 0.5)
        target_entropy = max(0.5, min(4.0, target_entropy))  # Clamp to realistic range
        
        payload_bytes = []
        
        for i in range(8):  # 8-byte CAN payload
            if target_entropy > 2.5:
                # Higher entropy - more random data
                byte_val = np.random.randint(0, 256)
            elif target_entropy > 1.5:
                # Medium entropy - some patterns
                if i < 4:
                    byte_val = np.random.choice([0x00, 0xFF, 0x80, np.random.randint(0, 256)], 
                                              p=[0.3, 0.2, 0.1, 0.4])
                else:
                    byte_val = np.random.randint(0, 256)
            else:
                # Lower entropy - more structured data
                if i < 2:
                    byte_val = np.random.choice([0x00, 0xFF], p=[0.7, 0.3])
                else:
                    byte_val = np.random.choice([0x00, 0xFF, np.random.randint(0, 256)], 
                                              p=[0.4, 0.3, 0.3])
            
            payload_bytes.append(f"{byte_val:02X}")
        
        return ' '.join(payload_bytes)

def main():
    """Main function to extract and synthesize normal CAN traffic"""
    
    print("CAN Bus Normal Traffic Extraction & Synthesis")
    print("=" * 50)
    
    extractor = NormalTrafficExtractor()
    
    # Step 1: Extract normal traffic from labeled datasets
    normal_features = extractor.extract_from_labeled_features()
    
    if normal_features is not None:
        # Step 2: Analyze normal traffic patterns
        analysis = extractor.analyze_normal_patterns(normal_features)
        
        # Step 3: Synthesize realistic normal raw traffic
        synthetic_normal = extractor.synthesize_normal_raw_traffic(normal_features, 50000)
        
        if synthetic_normal is not None:
            print("\n" + "="*60)
            print("SUCCESS: Normal traffic extraction and synthesis complete!")
            print("="*60)
            print("Generated Files:")
            print("- extracted_normal_features.csv (labeled normal features)")
            print("- synthesized_normal_traffic.csv (realistic normal CAN messages)")
            print("\nThis synthesized normal traffic can now be used as a")
            print("realistic baseline for generating sophisticated Phantom ECU attacks.")
        else:
            print("Error: Failed to synthesize normal traffic")
    else:
        print("Error: Failed to extract normal traffic features")

if __name__ == "__main__":
    main()
