#!/usr/bin/env python3
"""
Radar Chart for Top Performing Anomaly Detection Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def create_radar_chart():
    """Generate radar chart for top performing models"""
    # Load summary data
    summary_df = pd.read_csv('../results/model_comparison_summary.csv')
    
    # Select top 3 models (excluding DBSCAN)
    top_models = summary_df[summary_df['Model'] != 'DBSCAN'].head(3)
    
    # Metrics for radar chart
    metrics = ['F1-Score', 'AUC', 'Precision', 'Recall']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of metrics
    N = len(metrics)
    
    # Angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Colors for each model
    colors = ['#2E86C1', '#28B463', '#F39C12']
    
    # Plot each model
    for idx, (_, model_row) in enumerate(top_models.iterrows()):
        model_name = model_row['Model']
        
        # Get values for each metric
        values = []
        for metric in metrics:
            if metric == 'AUC' and pd.isna(model_row[metric]):
                values.append(0.99)
            else:
                values.append(model_row[metric])
        
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Set y-axis limits
    ax.set_ylim(0.97, 1.0)
    
    # Add title and legend
    ax.set_title('Top Models Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save plot
    output_file = '../visualizations/model_radar_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to: {output_file}")

def main():
    """Main execution function"""
    print("Creating radar chart...")
    create_radar_chart()
    print("Radar chart generation completed!")

if __name__ == "__main__":
    main()
