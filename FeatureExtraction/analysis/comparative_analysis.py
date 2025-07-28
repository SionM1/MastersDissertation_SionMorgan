#!/usr/bin/env python3
"""
Comparative Analysis and Visualization for CAN IDS Anomaly Detection Models
Generates performance comparison plots and clean summary tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """Generate comparative analysis and visualizations for anomaly detection models"""
    
    def __init__(self):
        self.results_dir = Path('results')
        self.hyperparams_dir = Path('hyperparameters')
        self.output_plots = []
        
    def load_results(self):
        """Load results from hyperparameter tuning and main evaluation"""
        # Load hyperparameter summary
        hyperparam_file = self.hyperparams_dir / 'hyperparameter_summary.csv'
        if hyperparam_file.exists():
            self.hyperparam_results = pd.read_csv(hyperparam_file)
        else:
            print("Warning: hyperparameter_summary.csv not found")
            self.hyperparam_results = None
            
        # Load main evaluation results
        main_results_file = self.results_dir / 'anomaly_detection_results.csv'
        if main_results_file.exists():
            self.main_results = pd.read_csv(main_results_file)
        else:
            print("Warning: anomaly_detection_results.csv not found")
            self.main_results = None
    
    def create_summary_csv(self):
        """Create clean model comparison summary CSV"""
        if self.main_results is None:
            print("No main results available for summary")
            return
            
        # Create clean summary
        summary_data = []
        
        for _, row in self.main_results.iterrows():
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
        
        # Sort by F1-Score descending
        summary_df = summary_df.sort_values('F1-Score', ascending=False)
        
        # Save to CSV
        output_file = 'model_comparison_summary.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
        
        return summary_df
    
    def plot_performance_metrics(self, summary_df):
        """Generate bar charts for key performance metrics"""
        # Filter out DBSCAN for better visualization (poor performance)
        df_filtered = summary_df[summary_df['Model'] != 'DBSCAN'].copy()
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Colors for consistent visualization
        colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD']
        
        # Plot 1: F1-Score
        axes[0, 0].bar(df_filtered['Model'], df_filtered['F1-Score'], color=colors[:len(df_filtered)])
        axes[0, 0].set_title('F1-Score Comparison')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].set_ylim(0.97, 1.0)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(df_filtered['F1-Score']):
            axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: AUC
        auc_data = df_filtered.dropna(subset=['AUC'])
        axes[0, 1].bar(auc_data['Model'], auc_data['AUC'], color=colors[:len(auc_data)])
        axes[0, 1].set_title('AUC Comparison')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_ylim(0.97, 1.0)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(auc_data['AUC']):
            axes[0, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Training Time
        axes[1, 0].bar(df_filtered['Model'], df_filtered['Training Time (s)'], color=colors[:len(df_filtered)])
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(df_filtered['Training Time (s)']):
            axes[1, 0].text(i, v + max(df_filtered['Training Time (s)']) * 0.01, 
                           f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Inference Time (log scale for better visibility)
        axes[1, 1].bar(df_filtered['Model'], df_filtered['Inference Time (s)'], color=colors[:len(df_filtered)])
        axes[1, 1].set_title('Inference Time Comparison')
        axes[1, 1].set_ylabel('Inference Time (seconds)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(df_filtered['Inference Time (s)']):
            axes[1, 1].text(i, v * 1.5, f'{v:.3f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'model_comparison_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        self.output_plots.append(output_file)
        print(f"Performance plots saved to: {output_file}")
        
    def plot_radar_chart(self, summary_df):
        """Generate radar chart for top performing models"""
        # Select top 3 models by F1-Score (excluding DBSCAN)
        top_models = summary_df[summary_df['Model'] != 'DBSCAN'].head(3)
        
        if len(top_models) < 2:
            print("Insufficient models for radar chart")
            return
            
        # Prepare metrics for radar chart (normalize to 0-1 scale)
        metrics = ['F1-Score', 'AUC', 'Precision', 'Recall']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of metrics
        N = len(metrics)
        
        # Angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Colors for each model
        colors = ['#2E86C1', '#28B463', '#F39C12']
        
        # Plot each model
        for idx, (_, model_row) in enumerate(top_models.iterrows()):
            model_name = model_row['Model']
            
            # Get values for each metric
            values = []
            for metric in metrics:
                if metric == 'AUC' and pd.isna(model_row[metric]):
                    values.append(0.99)  # Default for missing AUC
                else:
                    values.append(model_row[metric])
            
            values += values[:1]  # Complete the circle
            
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
        output_file = 'model_radar_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        self.output_plots.append(output_file)
        print(f"Radar chart saved to: {output_file}")
    
    def plot_training_vs_performance(self, summary_df):
        """Generate scatter plot of training time vs performance"""
        df_filtered = summary_df[summary_df['Model'] != 'DBSCAN'].copy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(df_filtered['Training Time (s)'], df_filtered['F1-Score'], 
                           s=200, alpha=0.7, c=range(len(df_filtered)), cmap='viridis')
        
        # Add model labels
        for idx, row in df_filtered.iterrows():
            ax.annotate(row['Model'], 
                       (row['Training Time (s)'], row['F1-Score']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('F1-Score')
        ax.set_title('Training Time vs Performance Trade-off')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        output_file = 'training_time_vs_performance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        self.output_plots.append(output_file)
        print(f"Training time comparison saved to: {output_file}")
    
    def generate_performance_table(self, summary_df):
        """Generate a clean performance table image"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Filter and format data for table
        table_data = summary_df.copy()
        
        # Format numeric columns
        for col in ['F1-Score', 'AUC', 'Precision', 'Recall']:
            if col in table_data.columns:
                table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        
        for col in ['Training Time (s)', 'Inference Time (s)']:
            if col in table_data.columns:
                table_data[col] = table_data[col].apply(lambda x: f'{x:.3f}')
        
        # Create table
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best performers
        for i in range(1, len(table_data) + 1):
            if table_data.iloc[i-1]['Model'] == 'Autoencoder':  # Best F1-Score
                for j in range(len(table_data.columns)):
                    table[(i, j)].set_facecolor('#E8F5E8')
        
        plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
        
        # Save plot
        output_file = 'model_performance_table.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        self.output_plots.append(output_file)
        print(f"Performance table saved to: {output_file}")
    
    def run_analysis(self):
        """Execute complete comparative analysis"""
        print("Starting comparative analysis...")
        
        # Load results
        self.load_results()
        
        # Create summary CSV
        summary_df = self.create_summary_csv()
        
        if summary_df is not None:
            # Generate all visualizations
            self.plot_performance_metrics(summary_df)
            self.plot_radar_chart(summary_df)
            self.plot_training_vs_performance(summary_df)
            self.generate_performance_table(summary_df)
            
            print(f"\nComparative analysis completed!")
            print(f"Generated files:")
            print(f"  - model_comparison_summary.csv")
            for plot in self.output_plots:
                print(f"  - {plot}")
        else:
            print("No results available for analysis")

def main():
    """Main execution function"""
    comparator = ModelComparator()
    comparator.run_analysis()

if __name__ == "__main__":
    main()
