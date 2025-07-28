#!/usr/bin/env python3
"""
Setup Verification Script
Tests that all components of the CAN Bus IDS project are working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("  [OK] pandas")
    except ImportError:
        print("  [FAIL] pandas - run: pip install pandas")
        return False
        
    try:
        import numpy as np
        print("  [OK] numpy")
    except ImportError:
        print("  [FAIL] numpy - run: pip install numpy")
        return False
        
    try:
        import sklearn
        print("  [OK] scikit-learn")
    except ImportError:
        print("  [FAIL] scikit-learn - run: pip install scikit-learn")
        return False
        
    try:
        import torch
        print("  [OK] torch")
    except ImportError:
        print("  [FAIL] torch - run: pip install torch")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print("  [OK] matplotlib")
    except ImportError:
        print("  [FAIL] matplotlib - run: pip install matplotlib")
        return False
        
    return True

def test_directory_structure():
    """Test that all required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "FeatureExtraction",
        "FeatureExtraction/data",
        "FeatureExtraction/models", 
        "FeatureExtraction/results",
        "FeatureExtraction/analysis",
        "FeatureExtraction/visualizations",
        "FeatureExtraction/scripts",
        "FeatureExtraction/hyperparameters",
        "AttackData",
        "StandardizedData"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  [OK] {directory}")
        else:
            print(f"  [FAIL] {directory} - missing")
            all_exist = False
            
    return all_exist

def test_key_files():
    """Test that key files exist"""
    print("\nTesting key files...")
    
    key_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "FeatureExtraction/README.md",
        "FeatureExtraction/anomaly_detection_models.py",
        "FeatureExtraction/analysis/simple_analysis.py"
    ]
    
    all_exist = True
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - missing")
            all_exist = False
            
    return all_exist

def test_data_files():
    """Test that data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        "FeatureExtraction/data/features_normal.csv",
        "FeatureExtraction/data/features_combined.csv"
    ]
    
    data_exists = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - needed for running analysis")
            data_exists = False
            
    return data_exists

def main():
    """Run all tests"""
    print("CAN Bus IDS Setup Verification")
    print("=" * 40)
    
    # Run all tests
    imports_ok = test_imports()
    dirs_ok = test_directory_structure()
    files_ok = test_key_files()
    data_ok = test_data_files()
    
    # Summary
    print("\n" + "=" * 40)
    print("VERIFICATION SUMMARY")
    print("=" * 40)
    
    if imports_ok:
        print("[OK] Package imports: PASS")
    else:
        print("[FAIL] Package imports: FAIL - install missing packages")
        
    if dirs_ok:
        print("[OK] Directory structure: PASS")
    else:
        print("[FAIL] Directory structure: FAIL - missing directories")
        
    if files_ok:
        print("[OK] Key files: PASS") 
    else:
        print("[FAIL] Key files: FAIL - missing files")
        
    if data_ok:
        print("[OK] Data files: PASS")
    else:
        print("[FAIL] Data files: FAIL - run feature extraction first")
    
    # Overall result
    if all([imports_ok, dirs_ok, files_ok]):
        print("\nSETUP VERIFICATION SUCCESSFUL!")
        print("Your project is ready for GitHub!")
        if not data_ok:
            print("\nNote: Data files missing - run feature extraction to generate them")
        return True
    else:
        print("\nSETUP VERIFICATION FAILED")
        print("Please fix the issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
