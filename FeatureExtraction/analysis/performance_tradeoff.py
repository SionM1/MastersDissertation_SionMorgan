#!/usr/bin/env python3
"""
Performance vs Speed Trade-off Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def create_tradeoff_plot():
    """Generate performance vs speed trade-off visualization"""
    # Load summary data
    summary_df = pd.read_csv('../results/model_comparison_summary.csv')
    
    # Filter out DBSCAN for better visualization
    df_filtered = summary_df[summary_df['Model'] != 'DBSCAN'].copy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training Time vs F1-Score
    colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']
    scatter1 = ax1.scatter(df_filtered['Training Time (s)'], df_filtered['F1-Score'], 
                          s=200, alpha=0.7, c=colors[:len(df_filtered)])
    
    # Add model labels
    for idx, row in df_filtered.iterrows():
        ax1.annotate(row['Model'], 
                    (row['Training Time (s)'], row['F1-Score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Training Time vs F1-Score')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inference Time vs F1-Score (log scale for inference time)
    scatter2 = ax2.scatter(df_filtered['Inference Time (s)'], df_filtered['F1-Score'], 
                          s=200, alpha=0.7, c=colors[:len(df_filtered)])
    
    # Add model labels
    for idx, row in df_filtered.iterrows():
        ax2.annotate(row['Model'], 
                    (row['Inference Time (s)'], row['F1-Score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Inference Time (seconds)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Inference Time vs F1-Score')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = '../visualizations/performance_tradeoff_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trade-off analysis saved to: {output_file}")

def main():
    """Main execution function"""
    print("Creating performance trade-off analysis...")
    create_tradeoff_plot()
    print("Trade-off analysis completed!")

if __name__ == "__main__":
    main()
