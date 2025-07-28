#!/usr/bin/env python3
"""
Convert Real RPM Data to Standard 4-Column Format
This script helps convert your raw CAN log to the standard format needed for analysis.
"""

import pandas as pd
import re
import os

def convert_rpm_log_to_standard(input_path, output_path):
    """
    Convert raw CAN log to standard format: Timestamp, CAN_ID, DLC, Payload
    
    Modify this function based on your actual log format!
    """
    print(f"Converting {input_path} to standard format...")
    
    data = []
    
    try:
        with open(input_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                # MODIFY THIS REGEX BASED ON YOUR LOG FORMAT
                # Current format assumes: Timestamp: X.X ID: XXX ... DLC: X payload_data
                match = re.search(r"Timestamp:\s*([\d.]+)\s+ID:\s*([\da-fA-F]+).*?DLC:\s*(\d+)\s+(.*)", line)
                
                if match:
                    try:
                        timestamp = float(match.group(1))
                        can_id = match.group(2).upper().zfill(8)  # Pad to 8 characters, uppercase
                        dlc = int(match.group(3))
                        payload = match.group(4).strip()
                        
                        # Clean up payload format - ensure space-separated hex bytes
                        payload_parts = payload.split()
                        if len(payload_parts) > 0:
                            # Ensure each part is proper 2-digit hex
                            clean_payload = []
                            for part in payload_parts:
                                try:
                                    # Convert to int then back to 2-digit hex
                                    val = int(part, 16) if '0x' not in part.lower() else int(part, 16)
                                    clean_payload.append(f"{val:02X}")
                                except:
                                    clean_payload.append("00")
                            payload = " ".join(clean_payload)
                        
                        data.append({
                            'Timestamp': timestamp,
                            'CAN_ID': can_id,
                            'DLC': dlc,
                            'Payload': payload
                        })
                        
                    except ValueError as e:
                        print(f"Warning: Could not parse line {line_num}: {line} - Error: {e}")
                        
        if data:
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values('Timestamp').reset_index(drop=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            print(f"Successfully converted {len(df)} records")
            print(f"Output saved to: {output_path}")
            
            # Show sample data
            print(f"\nSample of converted data:")
            print(df.head())
            
            # Show statistics
            print(f"\n📈 Conversion Statistics:")
            print(f"   • Time range: {df['Timestamp'].min():.3f} - {df['Timestamp'].max():.3f} seconds")
            print(f"   • Unique CAN IDs: {df['CAN_ID'].nunique()}")
            print(f"   • Most common CAN IDs:")
            id_counts = df['CAN_ID'].value_counts().head()
            for can_id, count in id_counts.items():
                print(f"     - {can_id}: {count} messages")
            
            return df
        else:
            print("No data was parsed from the input file")
            print("Please check your log format and update the regex pattern in this script")
            return None
            
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def detect_log_format(file_path):
    """
    Try to detect the format of your log file
    """
    print(f"🔍 Analyzing log format in {file_path}...")
    
    try:
        with open(file_path, 'r') as file:
            # Read first 10 lines to analyze format
            lines = [next(file).strip() for _ in range(min(10, sum(1 for _ in file)))]
            
        print(f"📝 First few lines of your log file:")
        for i, line in enumerate(lines[:5], 1):
            print(f"   {i}: {line}")
        
        # Try to detect common patterns
        patterns = [
            (r"Timestamp:\s*([\d.]+)\s+ID:\s*([\da-fA-F]+)", "Standard CAN log format"),
            (r"(\d+\.\d+)\s+([\da-fA-F]+)\s+(\d+)", "Time ID DLC format"),
            (r"(\d+)\s+([\da-fA-F]+)", "Simple Time ID format"),
        ]
        
        print(f"\n🔍 Pattern analysis:")
        for pattern, description in patterns:
            matches = sum(1 for line in lines if re.search(pattern, line))
            print(f"   • {description}: {matches}/{len(lines)} lines match")
        
    except Exception as e:
        print(f"Could not analyze file: {e}")

def main():
    """
    Main function - update the file paths here
    """
    print("Real RPM Data Converter")
    print("=" * 50)
    
    # UPDATE THESE PATHS TO MATCH YOUR FILES
    input_file = "raw_rpm_log.txt"      # Your raw log file
    output_file = "real_rpm_data.csv"    # Output standardized CSV
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f" Input file not found: {input_file}")
        print(f"\n Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith(('.txt', '.log', '.csv')):
                print(f"   • {file}")
        print(f"\n  Please update the 'input_file' variable in this script to point to your actual log file")
        return
    
    # Analyze the format first
    detect_log_format(input_file)
    
    # Convert the file
    df = convert_rpm_log_to_standard(input_file, output_file)
    
    if df is not None:
        print(f"\n Conversion complete! Next steps:")
        print(f"   1. Review the output file: {output_file}")
        print(f"   2. Run the integration pipeline: python integrate_real_rpm_data.py")
        print(f"   3. The pipeline will extract features and evaluate with Isolation Forest")
    else:
        print(f"\n Conversion failed. Please:")
        print(f"   1. Check your log file format")
        print(f"   2. Update the regex pattern in this script if needed")
        print(f"   3. Ensure your log contains CAN messages with timestamps")

if __name__ == "__main__":
    main()
