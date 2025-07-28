import pandas as pd
import os

def check_csv_columns(filepath):
    """Check the columns in a CSV file"""
    try:
        df = pd.read_csv(filepath, nrows=5)  # Just read first 5 rows
        print(f"\n{filepath}:")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("Sample data:")
        print(df.head(2))
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

print("="*60)
print("CHECKING CSV COLUMN NAMES")
print("="*60)

# Check normal dataset
normal_file = '../DatasetCVSConversion/normal_run_data.csv'
if os.path.exists(normal_file):
    normal_cols = check_csv_columns(normal_file)
else:
    print(f"Normal file not found: {normal_file}")

print("\n" + "="*60)
print("CHECKING ATTACK DATASETS")
print("="*60)

# Check attack datasets
attack_dir = '../AttackData'
if os.path.exists(attack_dir):
    for filename in os.listdir(attack_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(attack_dir, filename)
            check_csv_columns(filepath)
else:
    print(f"Attack directory not found: {attack_dir}")