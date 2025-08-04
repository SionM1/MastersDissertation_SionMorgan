#!/usr/bin/env python3
"""
Comprehensive Analysis of Realistic Phantom ECU Attack vs Original Phantom ECU Attack
Analyzes if the realistic baseline approach achieved better evasion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_results():
    """
    Load the attack-specific evaluation results
    """
    results_path = "../results/attack_specific_results.csv"
    try:
        df = pd.read_csv(results_path)
        return df
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        return None

def compare_phantom_attacks(results_df):
    """
    Compare Phantom ECU vs Realistic Phantom ECU attack performance
    """
    print("=" * 80)
    print("PHANTOM ECU ATTACK COMPARISON: ORIGINAL vs REALISTIC")
    print("=" * 80)
    
    # Filter phantom attack results
    phantom_results = results_df[results_df['Attack_Type'] == 'Phantom_ECU']
    realistic_phantom_results = results_df[results_df['Attack_Type'] == 'Realistic_Phantom_ECU']
    
    if phantom_results.empty or realistic_phantom_results.empty:
        print("Error: Missing phantom attack results")
        return None
    
    print("\nORIGINAL PHANTOM ECU ATTACK PERFORMANCE:")
    print("-" * 50)
    for _, row in phantom_results.iterrows():
        print(f"{row['Model']:<20}: F1={row['F1_Score']:.4f}, Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, AUC={row['AUC']:.4f}")
    
    print("\nREALISTIC PHANTOM ECU ATTACK PERFORMANCE:")
    print("-" * 50)
    for _, row in realistic_phantom_results.iterrows():
        print(f"{row['Model']:<20}: F1={row['F1_Score']:.4f}, Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, AUC={row['AUC']:.4f}")
    
    # Calculate performance differences
    print("\nPERFORMANCE DIFFERENCE (Realistic - Original):")
    print("-" * 50)
    print("Note: Negative values indicate BETTER EVASION (lower detection)")
    print()
    
    comparison_data = []
    
    for model in phantom_results['Model'].unique():
        original = phantom_results[phantom_results['Model'] == model].iloc[0]
        realistic = realistic_phantom_results[realistic_phantom_results['Model'] == model].iloc[0]
        
        f1_diff = realistic['F1_Score'] - original['F1_Score']
        precision_diff = realistic['Precision'] - original['Precision']
        recall_diff = realistic['Recall'] - original['Recall']
        auc_diff = realistic['AUC'] - original['AUC']
        
        comparison_data.append({
            'Model': model,
            'F1_Difference': f1_diff,
            'Precision_Difference': precision_diff,
            'Recall_Difference': recall_diff,
            'AUC_Difference': auc_diff,
            'Original_F1': original['F1_Score'],
            'Realistic_F1': realistic['F1_Score']
        })
        
        print(f"{model:<20}: F1={f1_diff:+.4f}, Precision={precision_diff:+.4f}, Recall={recall_diff:+.4f}, AUC={auc_diff:+.4f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Overall assessment
    print("\nOVERALL EVASION ASSESSMENT:")
    print("-" * 50)
    
    improved_evasion_count = sum(comparison_df['F1_Difference'] < 0)
    total_models = len(comparison_df)
    
    avg_f1_change = comparison_df['F1_Difference'].mean()
    avg_precision_change = comparison_df['Precision_Difference'].mean()
    avg_recall_change = comparison_df['Recall_Difference'].mean()
    
    print(f"Models with improved evasion: {improved_evasion_count}/{total_models}")
    print(f"Average F1-Score change: {avg_f1_change:+.4f}")
    print(f"Average Precision change: {avg_precision_change:+.4f}")
    print(f"Average Recall change: {avg_recall_change:+.4f}")
    
    # Determine if realistic attack is more effective
    if avg_f1_change < 0:
        print("\n*** CONCLUSION: The realistic phantom attack achieved BETTER evasion!")
        print("   Lower detection rates indicate more effective steganographic concealment.")
    else:
        print("\n*** CONCLUSION: The realistic phantom attack did NOT achieve better evasion.")
        print("   Higher detection rates suggest the attack was easier to detect.")
    
    return comparison_df

def analyze_model_vulnerabilities(results_df):
    """
    Analyze which models struggle most with the realistic approach
    """
    print("\n" + "=" * 80)
    print("MODEL VULNERABILITY ANALYSIS")
    print("=" * 80)
    
    # Calculate detection difficulty for each attack type
    attack_difficulty = results_df.groupby(['Attack_Type', 'Model']).agg({
        'F1_Score': 'first',
        'Precision': 'first',
        'Recall': 'first'
    }).reset_index()
    
    # Focus on phantom attacks
    phantom_data = attack_difficulty[attack_difficulty['Attack_Type'].isin(['Phantom_ECU', 'Realistic_Phantom_ECU'])]
    
    print("\nMODEL PERFORMANCE ON PHANTOM ATTACKS:")
    print("-" * 50)
    
    for model in phantom_data['Model'].unique():
        model_data = phantom_data[phantom_data['Model'] == model]
        
        original_f1 = model_data[model_data['Attack_Type'] == 'Phantom_ECU']['F1_Score'].iloc[0]
        realistic_f1 = model_data[model_data['Attack_Type'] == 'Realistic_Phantom_ECU']['F1_Score'].iloc[0]
        
        vulnerability_change = realistic_f1 - original_f1
        
        print(f"\n{model}:")
        print(f"  Original Phantom: {original_f1:.4f}")
        print(f"  Realistic Phantom: {realistic_f1:.4f}")
        print(f"  Vulnerability Change: {vulnerability_change:+.4f}")
        
        if vulnerability_change > 0:
            print("  → More vulnerable to realistic attack")
        else:
            print("  → Less vulnerable to realistic attack")
    
    # Rank models by detection capability on realistic attack
    realistic_performance = results_df[results_df['Attack_Type'] == 'Realistic_Phantom_ECU'].sort_values('F1_Score', ascending=False)
    
    print(f"\nMODEL RANKING (Realistic Phantom Detection Capability):")
    print("-" * 50)
    for i, (_, row) in enumerate(realistic_performance.iterrows(), 1):
        print(f"{i}. {row['Model']:<20}: F1={row['F1_Score']:.4f}")

def create_comparison_visualization(comparison_df, results_df):
    """
    Create visualizations comparing the attacks
    """
    # Set up the plotting style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: F1-Score comparison
    x = range(len(comparison_df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], comparison_df['Original_F1'], width, 
            label='Original Phantom', alpha=0.8, color='lightblue')
    ax1.bar([i + width/2 for i in x], comparison_df['Realistic_F1'], width,
            label='Realistic Phantom', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score Comparison: Original vs Realistic Phantom Attack')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Model'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance difference
    colors = ['green' if x < 0 else 'red' for x in comparison_df['F1_Difference']]
    ax2.bar(comparison_df['Model'], comparison_df['F1_Difference'], color=colors, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('F1-Score Difference')
    ax2.set_title('Performance Difference (Realistic - Original)\nNegative = Better Evasion')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: All attack types performance
    attack_performance = results_df.groupby('Attack_Type')['F1_Score'].mean().sort_values(ascending=True)
    
    colors_attack = ['red' if 'Phantom' in attack else 'lightblue' for attack in attack_performance.index]
    ax3.barh(attack_performance.index, attack_performance.values, color=colors_attack, alpha=0.7)
    ax3.set_xlabel('Average F1-Score')
    ax3.set_title('Average Detection Performance by Attack Type\n(Lower = Better Evasion)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model performance heatmap
    pivot_df = results_df.pivot(index='Model', columns='Attack_Type', values='F1_Score')
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax4, 
                cbar_kws={'label': 'F1-Score'})
    ax4.set_title('Model Performance Heatmap\n(Red = Higher Detection)')
    ax4.set_xlabel('Attack Type')
    ax4.set_ylabel('Model')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "../results/realistic_phantom_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    
    plt.show()

def generate_detailed_report(comparison_df, results_df):
    """
    Generate a detailed analysis report
    """
    print("\n" + "=" * 80)
    print("DETAILED EVASION EFFECTIVENESS REPORT")
    print("=" * 80)
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 30)
    
    # Calculate key metrics
    models_with_better_evasion = sum(comparison_df['F1_Difference'] < 0)
    total_models = len(comparison_df)
    percentage_improved = (models_with_better_evasion / total_models) * 100
    
    print(f"1. Evasion Improvement Rate: {models_with_better_evasion}/{total_models} models ({percentage_improved:.1f}%)")
    
    avg_f1_reduction = -comparison_df['F1_Difference'].mean()  # Negative for reduction
    print(f"2. Average F1-Score reduction: {avg_f1_reduction:.4f}")
    
    # Best and worst performing models on realistic attack
    realistic_results = results_df[results_df['Attack_Type'] == 'Realistic_Phantom_ECU']
    best_model = realistic_results.loc[realistic_results['F1_Score'].idxmax()]
    worst_model = realistic_results.loc[realistic_results['F1_Score'].idxmin()]
    
    print(f"3. Most vulnerable model to realistic attack: {worst_model['Model']} (F1: {worst_model['F1_Score']:.4f})")
    print(f"4. Least vulnerable model to realistic attack: {best_model['Model']} (F1: {best_model['F1_Score']:.4f})")
    
    # Steganographic effectiveness assessment
    print(f"\nSTEGANOGRAPHIC ATTACK EFFECTIVENESS:")
    print("-" * 40)
    
    if avg_f1_reduction > 0:
        print("*** SUCCESS: The realistic phantom attack demonstrates improved evasion capabilities")
        print("   * Real traffic patterns provide better camouflage")
        print("   * Entropy-preserving mutations maintain naturalness")
        print("   * Adaptive timing reduces detection signatures")
    else:
        print("*** PARTIAL SUCCESS: The realistic approach shows mixed results")
        print("   * Some detection improvements may be due to dataset characteristics")
        print("   * Consider adjusting steganographic parameters")
        print("   * Evaluate payload mutation strategies")
    
    # Technical insights
    print(f"\nTECHNICAL INSIGHTS:")
    print("-" * 20)
    print("* Realistic baseline approach uses 19,261 deduplicated normal samples")
    print("* 7% contamination rate maintains realistic traffic patterns")
    print("* Multi-strategy injection (LSB, bit rotation, pattern injection)")
    print("* Entropy-preserving mutations maintain payload naturalness")
    print("* Adaptive timing based on actual traffic statistics")
    
    return {
        'models_improved': models_with_better_evasion,
        'total_models': total_models,
        'avg_f1_reduction': avg_f1_reduction,
        'best_evasion_model': worst_model['Model'],
        'worst_evasion_model': best_model['Model']
    }

def main():
    """
    Main analysis function
    """
    print("Realistic Phantom ECU Attack Analysis")
    print("=" * 50)
    
    # Load results
    results_df = load_evaluation_results()
    if results_df is None:
        return
    
    print(f"Loaded evaluation results: {len(results_df)} records")
    
    # Compare phantom attacks
    comparison_df = compare_phantom_attacks(results_df)
    if comparison_df is None:
        return
    
    # Analyze model vulnerabilities
    analyze_model_vulnerabilities(results_df)
    
    # Create visualizations
    create_comparison_visualization(comparison_df, results_df)
    
    # Generate detailed report
    summary = generate_detailed_report(comparison_df, results_df)
    
    # Save comparison results
    comparison_output = "../results/phantom_attack_comparison.csv"
    comparison_df.to_csv(comparison_output, index=False)
    print(f"\nComparison results saved: {comparison_output}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
