#!/usr/bin/env python3
"""
Attack-Specific Evaluation for Anomaly Detection Models
Evaluates all trained models on each attack type individually
"""

import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import torch

def load_trained_models():
    """
    Load all previously trained models
    """
    models = {}
    model_files = {
        'OneClassSVM': 'oneclasssvm_model.pkl',
        'LOF': 'lof_model.pkl', 
        'Autoencoder': 'autoencoder_model.pkl',
        'DBSCAN': 'dbscan_model.pkl',
        'EllipticEnvelope': 'ellipticenvelope_model.pkl'
    }
    
    print("Loading trained models...")
    for model_name, filename in model_files.items():
        try:
            with open(filename, 'rb') as f:
                model_info = pickle.load(f)
                models[model_name] = model_info
                print(f"   {model_name}: Loaded successfully")
        except Exception as e:
            print(f"   {model_name}: Failed to load - {e}")
    
    # Load scaler
    try:
        with open('anomaly_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            print(f"   Scaler: Loaded successfully")
    except Exception as e:
        print(f"   Scaler: Failed to load - {e}")
        scaler = RobustScaler()  # Create new scaler as fallback
    
    return models, scaler

def evaluate_model_on_attack(model_name, model_info, X_test_scaled, y_test):
    """
    Evaluate a single model on attack-specific test data
    """
    model = model_info['model']
    
    start_time = time.time()
    
    try:
        # Handle different model types
        if model_name == 'Autoencoder':
            # Autoencoder uses reconstruction error
            test_tensor = torch.FloatTensor(X_test_scaled)
            reconstruction_errors = model.get_reconstruction_error(test_tensor)
            
            # Use fixed threshold (can be optimized)
            threshold = np.percentile(reconstruction_errors, 75)  # 75th percentile
            y_pred = (reconstruction_errors > threshold).astype(int)
            anomaly_scores = reconstruction_errors
            
        elif model_name == 'DBSCAN':
            # DBSCAN: Use contamination rate approach
            train_outlier_rate = np.mean(model_info['train_labels'] == -1)
            np.random.seed(42)
            n_outliers = int(len(X_test_scaled) * train_outlier_rate)
            y_pred = np.zeros(len(X_test_scaled))
            outlier_indices = np.random.choice(len(X_test_scaled), n_outliers, replace=False)
            y_pred[outlier_indices] = 1
            y_pred = y_pred.astype(int)
            anomaly_scores = y_pred.astype(float)
            
        else:
            # Standard sklearn models (OneClassSVM, LOF, EllipticEnvelope)
            predictions = model.predict(X_test_scaled)
            y_pred = (predictions == -1).astype(int)
            
            # Get decision scores for AUC
            try:
                decision_scores = model.decision_function(X_test_scaled)
                anomaly_scores = -decision_scores  # Convert to anomaly scores
            except:
                anomaly_scores = y_pred.astype(float)
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y_test, anomaly_scores)
        except:
            auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'inference_time': inference_time
        }
        
    except Exception as e:
        print(f"      Error evaluating {model_name}: {e}")
        return None

def evaluate_on_attack_type(models, scaler, attack_type, test_file):
    """
    Evaluate all models on a specific attack type
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING ON {attack_type.upper()} ATTACKS")
    print(f"{'='*60}")
    
    # Load test data
    test_data = pd.read_csv(test_file)
    print(f"Loaded test data: {len(test_data):,} samples")
    
    # Prepare features and labels
    feature_cols = [col for col in test_data.columns if col != 'label']
    X_test = test_data[feature_cols].values
    y_test = (test_data['label'] != 'normal').astype(int)  # 1 for attack, 0 for normal
    
    print(f"Features: {len(feature_cols)}")
    print(f"Normal samples: {np.sum(y_test == 0):,}")
    print(f"Attack samples: {np.sum(y_test == 1):,}")
    print(f"Contamination rate: {np.mean(y_test)*100:.1f}%")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate each model
    results = {}
    print(f"\nEvaluating models...")
    
    for model_name, model_info in models.items():
        print(f"\n   {model_name}:")
        result = evaluate_model_on_attack(model_name, model_info, X_test_scaled, y_test)
        if result:
            results[model_name] = result
            print(f"      Precision: {result['precision']:.4f}")
            print(f"      Recall: {result['recall']:.4f}")
            print(f"      F1-Score: {result['f1']:.4f}")
            if result['auc']:
                print(f"      AUC: {result['auc']:.4f}")
            print(f"      Inference: {result['inference_time']:.3f}s")
    
    return results

def create_attack_comparison_report(all_results):
    """
    Create comprehensive comparison report across all attack types
    """
    print(f"\n{'='*80}")
    print("ATTACK-SPECIFIC PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for attack_type, attack_results in all_results.items():
        for model_name, metrics in attack_results.items():
            comparison_data.append({
                'Attack_Type': attack_type,
                'Model': model_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'AUC': metrics['auc'] if metrics['auc'] else 0,
                'Inference_Time': metrics['inference_time']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save detailed results
    comparison_df.to_csv('attack_specific_results.csv', index=False)
    print(f"Detailed results saved to: attack_specific_results.csv")
    
    # Create summary by model
    print(f"\nPERFORMANCE SUMMARY BY MODEL:")
    print("-" * 80)
    
    for model_name in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model_name]
        avg_f1 = model_data['F1_Score'].mean()
        avg_precision = model_data['Precision'].mean()
        avg_recall = model_data['Recall'].mean()
        avg_auc = model_data['AUC'].mean()
        
        print(f"\n{model_name}:")
        print(f"   Average F1-Score: {avg_f1:.4f}")
        print(f"   Average Precision: {avg_precision:.4f}")
        print(f"   Average Recall: {avg_recall:.4f}")
        print(f"   Average AUC: {avg_auc:.4f}")
        
        # Best/worst attack types for this model
        best_attack = model_data.loc[model_data['F1_Score'].idxmax(), 'Attack_Type']
        worst_attack = model_data.loc[model_data['F1_Score'].idxmin(), 'Attack_Type']
        print(f"   Best on: {best_attack} (F1: {model_data['F1_Score'].max():.4f})")
        print(f"   Worst on: {worst_attack} (F1: {model_data['F1_Score'].min():.4f})")
    
    # Create summary by attack type
    print(f"\nPERFORMANCE SUMMARY BY ATTACK TYPE:")
    print("-" * 80)
    
    for attack_type in comparison_df['Attack_Type'].unique():
        attack_data = comparison_df[comparison_df['Attack_Type'] == attack_type]
        
        print(f"\n{attack_type.upper()} Attacks:")
        
        # Best model for this attack type
        best_model = attack_data.loc[attack_data['F1_Score'].idxmax(), 'Model']
        best_f1 = attack_data['F1_Score'].max()
        print(f"   Best Model: {best_model} (F1: {best_f1:.4f})")
        
        # Show all models for this attack type
        for _, row in attack_data.iterrows():
            print(f"   {row['Model']}: F1={row['F1_Score']:.4f}, AUC={row['AUC']:.4f}")
    
    return comparison_df

def main():
    """
    Main execution function
    """
    print("Attack-Specific Model Evaluation")
    print("=" * 50)
    
    # Load trained models
    models, scaler = load_trained_models()
    
    if not models:
        print("No models loaded. Please run anomaly_detection_models.py first.")
        return
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # Define attack test files
    attack_files = {
        'DoS': 'test_normal_vs_dos.csv',
        'Fuzzy': 'test_normal_vs_fuzzy.csv',
        'Gear': 'test_normal_vs_gear.csv',
        'RPM': 'test_normal_vs_rpm.csv'
    }
    
    # Evaluate on each attack type
    all_results = {}
    
    for attack_type, test_file in attack_files.items():
        if os.path.exists(test_file):
            results = evaluate_on_attack_type(models, scaler, attack_type, test_file)
            all_results[attack_type] = results
        else:
            print(f"Warning: {test_file} not found, skipping {attack_type}")
    
    # Create comprehensive comparison report
    if all_results:
        comparison_df = create_attack_comparison_report(all_results)
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED!")
        print(f"{'='*80}")
        print("Files created:")
        print("   - attack_specific_results.csv (detailed results)")
        print("\nKey insights:")
        print("   - Per-attack performance comparison")
        print("   - Best model for each attack type")
        print("   - Model strengths and weaknesses")
    
    else:
        print("No attack evaluations completed.")

if __name__ == "__main__":
    import os
    main()
