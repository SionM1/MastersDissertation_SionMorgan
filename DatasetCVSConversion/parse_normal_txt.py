import pandas as pd
import re
import os

def parse_log_to_csv(input_path, output_path):
    """
    Parse CAN bus log file to CSV format
    Expected input format: lines containing Timestamp, ID, DLC, and payload data
    """
    data = []
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return
    
    with open(input_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Skip lines that only contain timestamps without CAN data
            if re.match(r"^Timestamp:\s*[\d.]+\s*$", line):
                continue
                
            # Updated regex pattern to handle CAN log formats
            # Look for: Timestamp: X.X ID: XXX ... DLC: X payload_data
            match = re.search(r"Timestamp:\s*([\d.]+)\s+ID:\s*([\da-fA-F]+).*?DLC:\s*(\d+)\s+(.*)", line)
            if match:
                try:
                    timestamp = float(match.group(1))
                    can_id = match.group(2).upper()  # Ensure uppercase for consistency
                    dlc = int(match.group(3))
                    payload = match.group(4).strip()
                    
                    data.append({
                        'Timestamp': timestamp,
                        'CAN_ID': can_id,
                        'DLC': dlc,
                        'Payload': payload
                    })
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    print(f"Error: {e}")
            else:
                # Only show warning for lines that seem to have CAN data but don't match
                if "ID:" in line and "DLC:" in line:
                    print(f"Warning: Line {line_num} doesn't match expected format: {line}")
    
    if not data:
        print("No data was parsed from the input file")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully parsed {len(df)} rows and saved to {output_path}")
    
    # Display first few rows for verification
    print("\nFirst 5 rows of parsed data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Parse the normal run data
    input_file = "normal_run_data.txt"
    output_file = "../DatasetCVSConversion/normal_run_data.csv"  # Save to DatasetCVSConversion directory
    
    parse_log_to_csv(input_file, output_file)

