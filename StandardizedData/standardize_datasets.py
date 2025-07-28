import pandas as pd
import os

def convert_attack_dataset_to_standard(filepath, output_filepath):
    """Convert attack dataset to standard format"""
    print(f"Converting {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Get column names
        cols = df.columns.tolist()
        
        # Create standardized dataset
        standard_df = pd.DataFrame()
        
        # Timestamp (first column)
        standard_df['Timestamp'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        
        # CAN ID (second column) - ensure it's in hex format
        standard_df['CAN_ID'] = df.iloc[:, 1].astype(str).str.upper()
        
        # DLC (third column)  
        standard_df['DLC'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        
        # Combine payload bytes (columns 3 to -2, excluding last 'R' column)
        payload_cols = cols[3:-1] if cols[-1] == 'R' else cols[3:]
        
        def combine_payload_bytes(row):
            payload_parts = []
            for col in payload_cols:
                val = str(row[col]) if pd.notna(row[col]) else '00'
                # Clean up the value - remove decimal points, handle 'nan'
                val = val.split('.')[0] if '.' in val else val
                val = val.replace('nan', '00')
                
                # Convert to proper hex format
                try:
                    if val.isdigit():
                        val = f"{int(val):02x}"
                    elif val.lower() in ['nan', 'none', '']:
                        val = '00'
                    else:
                        # Assume it's already hex, just format it
                        val = f"{int(val, 16):02x}"
                except:
                    val = '00'
                    
                payload_parts.append(val)
            
            return ' '.join(payload_parts).upper()
        
        standard_df['Payload'] = df.apply(combine_payload_bytes, axis=1)
        
        # Remove any rows with invalid data
        standard_df = standard_df.dropna(subset=['Timestamp', 'CAN_ID'])
        
        # Save converted dataset
        standard_df.to_csv(output_filepath, index=False)
        print(f"Converted dataset saved: {output_filepath}")
        print(f"Final shape: {standard_df.shape}")
        print("Sample converted data:")
        print(standard_df.head(3))
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"Error converting {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("STANDARDIZING ALL DATASETS TO 4-COLUMN FORMAT")
    print("="*60)
    
    # Create output directory for standardized datasets
    output_dir = '../StandardizedData'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if normal dataset already in correct format
    normal_file = '../DatasetCVSConversion/normal_run_data.csv'
    if os.path.exists(normal_file):
        print(f"Normal dataset is already in standard format")
        # Copy to standardized directory
        df = pd.read_csv(normal_file)
        df.to_csv(os.path.join(output_dir, 'normal_run_data.csv'), index=False)
        print(f"Normal dataset copied to {output_dir}")
    
    # Convert attack datasets
    attack_dir = '../AttackData'
    if os.path.exists(attack_dir):
        for filename in os.listdir(attack_dir):
            if filename.endswith('.csv'):
                input_filepath = os.path.join(attack_dir, filename)
                output_filepath = os.path.join(output_dir, filename)
                convert_attack_dataset_to_standard(input_filepath, output_filepath)
    
    print("\n" + "="*60)
    print("STANDARDIZATION COMPLETE!")
    print("="*60)
    print(f"All standardized datasets are in: {output_dir}")
    print("Now you can run feature extraction on the standardized datasets.")

if __name__ == "__main__":
    main()