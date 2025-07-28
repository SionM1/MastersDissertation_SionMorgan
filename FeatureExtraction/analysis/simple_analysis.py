#!/usr/bin/env python3
"""
Simple Comparative Analysis for CAN IDS Anomaly Detection Models
Generates performance comparison plots and clean summary tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

def create_summary_csv():
    """Create clean model comparison summary CSV from results"""
    results_file = Path('../results/anomaly_detection_results.csv')
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
        
    # Load results
    df = pd.read_csv(results_file)
    
    # Create clean summary
    summary_data = []
    
    for _, row in df.iterrows():
        summary_data.append({
            'Model': row['Model'],
            'F1-Score': round(row['F1-Score'], 4),
            'AUC': round(row['AUC'], 4) if pd.notna(row['AUC']) else None,
            'Precision': round(row['Precision'], 4),
            'Recall': round(row['Recall'], 4),
            'Training Time (s)': round(row['Training Time (s)'], 3),
            'Inference Time (s)': round(row['Inference Time (s)'], 3)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1-Score', ascending=False)
    
    # Save to CSV
    output_file = '../results/model_comparison_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}")
    
    return summary_df

def plot_performance_comparison(summary_df):
    """Generate performance comparison plots"""
    # Filter out DBSCAN for better visualization
    df_filtered = summary_df[summary_df['Model'] != 'DBSCAN'].copy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']
    
    # F1-Score comparison
    axes[0, 0].bar(df_filtered['Model'], df_filtered['F1-Score'], color=colors[:len(df_filtered)])
    axes[0, 0].set_title('F1-Score Comparison')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_ylim(0.97, 1.0)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # AUC comparison
    auc_data = df_filtered.dropna(subset=['AUC'])
    axes[0, 1].bar(auc_data['Model'], auc_data['AUC'], color=colors[:len(auc_data)])
    axes[0, 1].set_title('AUC Comparison')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_ylim(0.97, 1.0)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    axes[1, 0].bar(df_filtered['Model'], df_filtered['Training Time (s)'], color=colors[:len(df_filtered)])
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Inference time comparison (log scale)
    axes[1, 1].bar(df_filtered['Model'], df_filtered['Inference Time (s)'], color=colors[:len(df_filtered)])
    axes[1, 1].set_title('Inference Time Comparison')
    axes[1, 1].set_ylabel('Inference Time (seconds)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_file = '../visualizations/model_comparison_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance plots saved to: {output_file}")

def main():
    """Main execution function"""
    print("Starting comparative analysis...")
    
    # Create summary CSV
    summary_df = create_summary_csv()
    
    if summary_df is not None:
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Generate plots
        plot_performance_comparison(summary_df)
        
        print("\nComparative analysis completed!")
        print("Generated files:")
        print("  - results/model_comparison_summary.csv")
        print("  - visualizations/model_comparison_plots.png")
    else:
        print("No results available for analysis")

if __name__ == "__main__":
    main()
