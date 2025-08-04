#!/usr/bin/env python3
"""
Realistic Phantom ECU Attack Generator v2.0
Creates sophisticated steganographic CAN bus attacks using real normal traffic
patterns extracted from car hacking datasets for maximum realism and consistency.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

class RealisticPhantomECUGenerator:
    def __init__(self, normal_traffic_path='synthesized_normal_traffic.csv', contamination_rate=0.12):
        """
        Initialize realistic Phantom ECU attack generator
        
        Args:
            normal_traffic_path: Path to realistic normal CAN traffic
            contamination_rate: Attack contamination percentage (default 12%)
        """
        self.contamination_rate = contamination_rate
        self.normal_traffic = self.load_normal_traffic(normal_traffic_path)
        self.attack_messages = []
        
        # Advanced steganographic parameters based on real traffic analysis
        self.target_can_ids = ['0002', '0130', '0131', '0140', '0316', '0350', '043F', '0545']
        self.entropy_preservation_factor = 0.85  # Maintain 85% of original entropy
        self.timing_stealth_variance = 0.02  # 2% timing variance for stealth
        self.escalation_curve = 1.15  # Gradual escalation over time
        self.payload_mutation_probability = 0.25  # 25% chance of payload mutation
        
        # Real traffic pattern constraints
        self.normal_entropy_mean = 1.954042
        self.normal_entropy_std = 0.901086
        self.normal_time_delta_mean = 0.000594
        self.normal_time_delta_std = 0.000791
        
    def load_normal_traffic(self, traffic_path):
        """Load realistic normal CAN traffic"""
        try:
            if os.path.exists(traffic_path):
                df = pd.read_csv(traffic_path)
                print(f"Loaded {len(df):,} normal CAN messages from: {traffic_path}")
                return df
            else:
                print(f"Error: Normal traffic file not found: {traffic_path}")
                return None
        except Exception as e:
            print(f"Error loading normal traffic: {e}")
            return None
    
    def calculate_payload_entropy(self, payload_str):
        """Calculate Shannon entropy of a CAN payload"""
        if not payload_str or payload_str.strip() == '':
            return 0.0
        
        # Convert hex payload to bytes
        try:
            hex_bytes = payload_str.replace(' ', '')
            byte_values = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
        except:
            return 0.0
        
        # Calculate Shannon entropy
        if len(byte_values) == 0:
            return 0.0
        
        from collections import Counter
        counts = Counter(byte_values)
        probabilities = [count / len(byte_values) for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def generate_entropy_preserving_payload(self, original_payload, attack_intensity=1.0):
        """
        Generate steganographic payload that preserves entropy characteristics
        while injecting malicious data through sophisticated bit manipulation
        """
        payload_bytes = original_payload.split()
        modified_bytes = []
        
        # Calculate original entropy
        original_entropy = self.calculate_payload_entropy(original_payload)
        target_entropy = original_entropy * self.entropy_preservation_factor
        
        for i, byte_str in enumerate(payload_bytes):
            if byte_str and byte_str != '':
                try:
                    byte_val = int(byte_str, 16)
                    
                    # Apply sophisticated steganographic mutations
                    if random.random() < (self.payload_mutation_probability * attack_intensity):
                        
                        # Strategy 1: LSB steganography (maintains high-order structure)
                        if random.random() < 0.4:
                            # Flip 1-2 LSBs for data injection
                            lsb_mask = random.choice([0x01, 0x02, 0x03])
                            byte_val ^= lsb_mask
                        
                        # Strategy 2: Bit rotation (maintains bit count, changes pattern)
                        elif random.random() < 0.3:
                            # Rotate bits to inject steganographic patterns
                            rotation = random.choice([1, 2, 7])  # Subtle rotations
                            byte_val = ((byte_val << rotation) | (byte_val >> (8 - rotation))) & 0xFF
                        
                        # Strategy 3: Entropy-guided mutations (preserve statistical properties)
                        elif random.random() < 0.2:
                            # Add controlled noise that maintains entropy
                            noise = random.choice([-2, -1, 1, 2])  # Small value changes
                            byte_val = max(0, min(255, byte_val + noise))
                        
                        # Strategy 4: Pattern injection (for advanced attacks)
                        else:
                            # Inject subtle patterns that escalate over time
                            if attack_intensity > 1.3:
                                # More aggressive pattern injection
                                if i % 2 == 0:  # Even positions
                                    byte_val ^= 0x04  # Flip 3rd bit
                                else:  # Odd positions
                                    byte_val ^= 0x08  # Flip 4th bit
                            else:
                                # Subtle pattern injection
                                byte_val ^= (i % 4)  # Position-based pattern
                    
                    modified_bytes.append(f"{byte_val:02X}")
                    
                except ValueError:
                    modified_bytes.append(byte_str)
            else:
                modified_bytes.append("00")
        
        modified_payload = ' '.join(modified_bytes)
        
        # Verify entropy preservation
        new_entropy = self.calculate_payload_entropy(modified_payload)
        entropy_ratio = new_entropy / original_entropy if original_entropy > 0 else 1.0
        
        # If entropy changed too much, apply compensation
        if entropy_ratio < 0.7 or entropy_ratio > 1.4:
            # Fall back to more conservative mutation
            return self.apply_conservative_mutation(original_payload, attack_intensity)
        
        return modified_payload
    
    def apply_conservative_mutation(self, original_payload, attack_intensity):
        """Apply conservative mutations when entropy preservation fails"""
        payload_bytes = original_payload.split()
        modified_bytes = []
        
        for byte_str in payload_bytes:
            if byte_str and byte_str != '':
                try:
                    byte_val = int(byte_str, 16)
                    
                    # Only apply minimal LSB changes
                    if random.random() < (0.15 * attack_intensity):
                        byte_val ^= 0x01  # Flip only LSB
                    
                    modified_bytes.append(f"{byte_val:02X}")
                except ValueError:
                    modified_bytes.append(byte_str)
            else:
                modified_bytes.append("00")
        
        return ' '.join(modified_bytes)
    
    def calculate_adaptive_timing(self, base_timestamp, message_index, total_messages):
        """
        Calculate adaptive timing that maintains normal patterns while enabling stealth
        """
        # Base timing follows normal distribution
        normal_delta = np.random.normal(self.normal_time_delta_mean, self.normal_time_delta_std)
        normal_delta = max(0.0001, min(0.01, normal_delta))  # Clamp to realistic range
        
        # Add stealth variance
        stealth_variance = random.uniform(-self.timing_stealth_variance, self.timing_stealth_variance)
        
        # Progressive timing adjustment (very subtle)
        progress_factor = message_index / total_messages
        timing_drift = progress_factor * 0.0001  # Minimal drift over time
        
        # Calculate final timing
        final_delta = normal_delta * (1 + stealth_variance) + timing_drift
        
        return base_timestamp + final_delta
    
    def should_inject_attack(self, can_id, message_index, total_messages, original_payload):
        """
        Sophisticated attack injection decision based on multiple factors
        """
        # Must target specific CAN IDs
        if can_id not in self.target_can_ids:
            return False, 0.0
        
        # Calculate base attack probability
        base_probability = self.contamination_rate
        
        # Factor 1: Progressive escalation
        progress = message_index / total_messages
        escalation_multiplier = 1 + (progress * (self.escalation_curve - 1))
        
        # Factor 2: Payload entropy compatibility
        payload_entropy = self.calculate_payload_entropy(original_payload)
        entropy_factor = 1.0
        if payload_entropy > 2.5:  # High entropy payloads easier to hide in
            entropy_factor = 1.2
        elif payload_entropy < 1.0:  # Low entropy payloads more risky
            entropy_factor = 0.7
        
        # Factor 3: CAN ID priority (some IDs are safer targets)
        id_priority = {
            '0002': 1.2,  # System messages - higher stealth
            '0130': 1.0,  # Standard priority
            '0131': 1.0,
            '0140': 0.9,  # Slightly lower priority
            '0316': 1.1,
            '0350': 1.0,
            '043F': 0.8,  # Lower priority
            '0545': 1.1
        }
        id_factor = id_priority.get(can_id, 1.0)
        
        # Calculate final attack probability
        attack_probability = base_probability * escalation_multiplier * entropy_factor * id_factor
        attack_probability = min(attack_probability, 0.35)  # Cap at 35%
        
        # Determine attack intensity
        attack_intensity = escalation_multiplier
        
        return random.random() < attack_probability, attack_intensity
    
    def generate_realistic_phantom_attack(self, output_filename='Realistic_Phantom_ECU_dataset.csv'):
        """
        Generate sophisticated Phantom ECU attack using realistic normal traffic patterns
        """
        if self.normal_traffic is None:
            print("Error: No normal traffic data available")
            return None
        
        print(f"Generating Realistic Phantom ECU attack...")
        print(f"Base normal messages: {len(self.normal_traffic):,}")
        print(f"Target contamination: {self.contamination_rate:.1%}")
        
        attack_dataset = []
        total_messages = len(self.normal_traffic)
        attack_count = 0
        
        # Process each normal message
        for idx, row in self.normal_traffic.iterrows():
            timestamp = row['Timestamp']
            can_id = row['CAN_ID']
            dlc = row['DLC']
            payload = row['Payload']
            
            # Determine if attack should be injected
            should_attack, attack_intensity = self.should_inject_attack(
                can_id, idx, total_messages, payload
            )
            
            if should_attack:
                # Generate steganographic attack message
                malicious_payload = self.generate_entropy_preserving_payload(
                    payload, attack_intensity
                )
                
                # Calculate adaptive timing
                stealth_timestamp = self.calculate_adaptive_timing(
                    timestamp, idx, total_messages
                )
                
                attack_dataset.append({
                    'Timestamp': stealth_timestamp,
                    'CAN_ID': can_id,
                    'DLC': dlc,
                    'Payload': malicious_payload
                })
                attack_count += 1
                
            else:
                # Keep original normal message with slight timing adjustment
                stealth_timestamp = self.calculate_adaptive_timing(
                    timestamp, idx, total_messages
                )
                
                attack_dataset.append({
                    'Timestamp': stealth_timestamp,
                    'CAN_ID': can_id,
                    'DLC': dlc,
                    'Payload': payload
                })
        
        # Create final dataset
        attack_df = pd.DataFrame(attack_dataset)
        attack_df = attack_df.sort_values('Timestamp')  # Maintain chronological order
        
        # Save attack dataset
        attack_df.to_csv(output_filename, index=False)
        
        # Calculate statistics
        actual_contamination = attack_count / total_messages
        
        print(f"\n{'='*60}")
        print("REALISTIC PHANTOM ECU ATTACK GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output file: {output_filename}")
        print(f"Total messages: {total_messages:,}")
        print(f"Attack messages: {attack_count:,}")
        print(f"Target contamination: {self.contamination_rate:.2%}")
        print(f"Actual contamination: {actual_contamination:.2%}")
        print(f"Attack efficiency: {(actual_contamination/self.contamination_rate)*100:.1f}%")
        
        print(f"\nAttack Characteristics:")
        print(f"- Entropy-preserving payload mutations")
        print(f"- Adaptive timing based on real traffic patterns")
        print(f"- Progressive escalation (factor: {self.escalation_curve})")
        print(f"- Multi-strategy steganographic injection")
        print(f"- Real normal traffic baseline (19,261 samples)")
        
        return attack_df

def main():
    """Main function to generate realistic Phantom ECU attack"""
    
    print("Realistic Phantom ECU Attack Generator v2.0")
    print("Using real car hacking dataset normal traffic patterns")
    print("=" * 60)
    
    # Check if normal traffic exists
    normal_traffic_file = 'synthesized_normal_traffic.csv'
    if not os.path.exists(normal_traffic_file):
        print(f"Error: Normal traffic file not found: {normal_traffic_file}")
        print("Please run extract_normal_traffic.py first to generate normal traffic")
        return
    
    # Initialize generator with realistic parameters
    generator = RealisticPhantomECUGenerator(
        normal_traffic_path=normal_traffic_file,
        contamination_rate=0.12  # 12% contamination for realistic steganographic attack
    )
    
    # Generate the attack
    attack_data = generator.generate_realistic_phantom_attack('Realistic_Phantom_ECU_dataset.csv')
    
    if attack_data is not None:
        print(f"\n{'='*60}")
        print("SUCCESS: Realistic Phantom ECU attack generated!")
        print(f"{'='*60}")
        print("This attack uses:")
        print("- Real normal CAN traffic patterns from car hacking datasets")
        print("- Entropy-preserving steganographic payload mutations")
        print("- Adaptive timing based on actual traffic statistics")
        print("- Progressive escalation for advanced evasion")
        print("- Multi-strategy injection (LSB, bit rotation, pattern injection)")
        print("\nThis should provide a much more realistic and challenging")
        print("attack for evaluating your anomaly detection models.")
    else:
        print("Error: Failed to generate realistic Phantom ECU attack")

if __name__ == "__main__":
    main()
