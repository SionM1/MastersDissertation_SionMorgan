import pandas as pd
import numpy as np
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CANFeatureExtractor:
    def __init__(self, window_size=1.0):
        """
        window_size: Time window in seconds for frequency analysis
        """
        self.window_size = window_size
        
    def calculate_time_delta(self, df):
        """Calculate time deltas between consecutive messages"""
        df = df.sort_values('Timestamp').copy()
        df['time_delta'] = df['Timestamp'].diff().fillna(0)
        return df
    
    def calculate_message_frequency(self, df):
        """Calculate frequency of each CAN ID per time window"""
        df['time_window'] = (df['Timestamp'] // self.window_size).astype(int)
        
        # Count messages per ID per window
        freq_stats = []
        for window in df['time_window'].unique():
            window_data = df[df['time_window'] == window]
            id_counts = window_data['CAN_ID'].value_counts()
            
            for _, row in window_data.iterrows():
                can_id = row['CAN_ID']
                frequency = id_counts.get(can_id, 0)
                freq_stats.append({
                    'index': row.name,
                    'msg_frequency': frequency,
                    'unique_ids_in_window': len(id_counts),
                    'total_msgs_in_window': len(window_data)
                })
        
        freq_df = pd.DataFrame(freq_stats).set_index('index')
        return df.join(freq_df)
    
    def calculate_payload_entropy(self, df):
        """Calculate Shannon entropy of payload data"""
        def shannon_entropy(data_str):
            if pd.isna(data_str) or len(str(data_str)) == 0:
                return 0
            
            try:
                # Remove spaces and convert hex to bytes
                clean_data = str(data_str).replace(' ', '').replace('0x', '')
                if len(clean_data) % 2 != 0:
                    clean_data = '0' + clean_data
                byte_data = bytes.fromhex(clean_data)
                
                # Calculate frequency of each byte
                byte_counts = defaultdict(int)
                for byte in byte_data:
                    byte_counts[byte] += 1
                
                # Calculate Shannon entropy
                total_bytes = len(byte_data)
                entropy = 0
                for count in byte_counts.values():
                    probability = count / total_bytes
                    if probability > 0:
                        entropy -= probability * np.log2(probability)
                
                return entropy
            except:
                return 0
        
        df['payload_entropy'] = df['Payload'].apply(shannon_entropy)
        return df
    
    def extract_additional_features(self, df):
        """Extract additional statistical features"""
        # CAN ID statistics - convert hex to numeric with improved cleaning
        def hex_to_numeric(hex_str):
            try:
                # Strip whitespace and remove "0x" prefix for reliable conversion
                clean_hex = str(hex_str).strip().replace("0x", "").replace("0X", "")
                return int(clean_hex, 16)
            except:
                return 0
                
        df['can_id_numeric'] = df['CAN_ID'].apply(hex_to_numeric)
        df['can_id_std'] = df.groupby('time_window')['can_id_numeric'].transform('std').fillna(0)
        
        # Payload length
        df['payload_length'] = df['Payload'].astype(str).str.len()
            
        # Time delta statistics within window
        df['time_delta_mean'] = df.groupby('time_window')['time_delta'].transform('mean')
        df['time_delta_std'] = df.groupby('time_window')['time_delta'].transform('std').fillna(0)
        
        return df
    
    def process_dataset(self, filepath, label):
        """Process a single dataset file"""
        print(f"Processing {filepath}...")
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} records")
            
            # Verify standard format (all datasets should now have these columns)
            required_columns = ['Timestamp', 'CAN_ID', 'DLC', 'Payload']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must have columns: {required_columns}. Found: {list(df.columns)}")
            
            # Basic preprocessing
            df = self.calculate_time_delta(df)
            df = self.calculate_message_frequency(df)
            df = self.calculate_payload_entropy(df)
            df = self.extract_additional_features(df)
            
            # Select feature columns
            feature_columns = [
                'time_delta',
                'msg_frequency', 
                'unique_ids_in_window',
                'total_msgs_in_window',
                'payload_entropy',
                'can_id_std',
                'payload_length',
                'time_delta_mean',
                'time_delta_std'
            ]
            
            # Create feature dataset
            features_df = df[feature_columns].copy()
            features_df['label'] = label
            
            # Print feature column names after extraction
            print(f"Feature columns: {features_df.columns.tolist()}")
            print(f"Extracted features: {features_df.shape}")
            return features_df
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    extractor = CANFeatureExtractor(window_size=1.0)

    print("="*50)
    print("PROCESSING NORMAL TRAFFIC")
    print("="*50)

    normal_file = '../DatasetCVSConversion/normal_run_data.csv'
    if not os.path.exists(normal_file):
        print(f"Normal data file not found: {normal_file}")
        return

    normal_features = extractor.process_dataset(normal_file, 'normal')
    if normal_features is not None:
        normal_features.to_csv('features_normal.csv', index=False)
        print(f"Normal features saved: {len(normal_features)} samples")
        
        # Save feature column order to ensure consistency
        feature_order = normal_features.columns.tolist()
        with open('feature_order.txt', 'w') as f:
            for feature in feature_order:
                f.write(f"{feature}\n")
        print(f"Feature order saved to feature_order.txt: {len(feature_order)} columns")

    print("\n" + "="*50)
    print("PROCESSING ATTACK DATASETS")
    print("="*50)

    attack_datasets = []
    attack_dir = '../AttackData'
    if os.path.exists(attack_dir):
        for filename in os.listdir(attack_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(attack_dir, filename)
                attack_features = extractor.process_dataset(filepath, 'attack')
                if attack_features is not None:
                    attack_datasets.append(attack_features)

    if attack_datasets:
        combined_attacks = pd.concat(attack_datasets, ignore_index=True)
        combined_attacks.to_csv('features_attack.csv', index=False)
        print(f"Attack features saved: {len(combined_attacks)} samples")

        if normal_features is not None:
            all_features = pd.concat([normal_features, combined_attacks], ignore_index=True)
            all_features.to_csv('features_combined.csv', index=False)
            print(f"Combined dataset saved: {len(all_features)} samples")

            print("\nFeature extraction complete!")
            print(f"Normal samples: {len(normal_features)}")
            print(f"Attack samples: {len(combined_attacks)}")
            print(f"Total samples: {len(all_features)}")
            
            # Show label distribution
            print(f"\nLabel distribution:")
            print(all_features['label'].value_counts())
            
            # Print final feature summary
            print(f"\nFinal feature columns: {all_features.columns.tolist()}")
    else:
        print("No attack datasets found!")


if __name__ == "__main__":
    main()