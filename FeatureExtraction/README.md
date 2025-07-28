# FeatureExtraction Directory

This directory contains the feature extraction and anomaly detection system for CAN bus intrusion detection.

## Directory Structure

```
FeatureExtraction/
├── README.md                           # This file
├── anomaly_detection_models.py         # Main anomaly detection framework
├──   data/                               # Processed datasets
│   ├── features_normal.csv                # Normal traffic features
│   ├── features_combined.csv              # Combined normal + attack features  
│   ├── features_attack.csv                # Attack-only features
│   ├── features_dos.csv                   # DoS attack features
│   ├── features_fuzzy.csv                 # Fuzzy attack features
│   ├── test_normal_vs_dos.csv            # DoS vs normal test set
│   ├── test_normal_vs_fuzzy.csv          # Fuzzy vs normal test set
│   ├── test_normal_vs_gear.csv           # Gear vs normal test set
│   ├── test_normal_vs_rpm.csv            # RPM vs normal test set
│   └── feature_order.txt                  # Feature column order
├──  models/                             # Trained models and preprocessors
│   ├── anomaly_scaler.pkl                 # RobustScaler for anomaly detection
│   ├── autoencoder_model.pkl              # Trained Autoencoder model
│   ├── dbscan_model.pkl                   # Trained DBSCAN model
│   ├── ellipticenvelope_model.pkl         # Trained Elliptic Envelope model
│   ├── isolation_forest_model.pkl         # Trained Isolation Forest model
│   ├── label_encoder.pkl                  # Label encoder for classification
│   ├── lof_model.pkl                      # Trained LOF model
│   ├── oneclasssvm_model.pkl              # Trained One-Class SVM model
│   ├── random_forest_model.pkl            # Trained Random Forest model
│   └── robust_scaler.pkl                  # RobustScaler for classification
├──  results/                            # Evaluation results
│   ├── anomaly_detection_results.csv      # Anomaly detection performance
│   ├── attack_specific_results.csv        # Attack-specific evaluation
│   └── results.csv                        # General results
├──  evaluation/                         # Evaluation scripts
│   ├── create_attack_specific_evaluation.py  # Create attack-specific datasets
│   └── evaluate_attack_specific.py           # Attack-specific evaluation
├──  scripts/                            # Utility and training scripts
│   ├── feature_engineering.py             # Feature extraction from raw data
│   ├── train_models_new.py                # Model training pipeline
│   ├── split_attack_data.py               # Split attacks by type
│   └── debug_columns.py                   # Debug utility for columns
├──  hyperparameters/                    # Hyperparameter tuning
│   ├── hyperparameter_tuning.py           # Full hyperparameter tuning
│   ├── hyperparameter_tuning_fast.py      # Fast hyperparameter tuning
│   ├── hyperparameter_results.csv         # Detailed tuning results
│   ├── hyperparameter_summary.csv         # Best parameters summary
│   └── hyperparameter_results_analysis.md # Analysis and documentation
├──  analysis/                           # Comparative analysis scripts
│   ├── simple_analysis.py                 # Main analysis script
│   ├── radar_chart.py                     # Radar chart generator
│   ├── performance_tradeoff.py            # Trade-off analysis
│   └── comparative_analysis_summary.md    # Comprehensive analysis document
├──  visualizations/                     # Generated plots and charts
│   ├── model_comparison_plots.png         # Performance comparison bar charts
│   ├── model_radar_comparison.png         # Radar chart of top models
│   └── performance_tradeoff_analysis.png  # Speed vs performance plots
└── 📁 __pycache__/                        # Python cache files
```

## Quick Start

### 1. Main Anomaly Detection Framework
```bash
python anomaly_detection_models.py
```
Runs the complete anomaly detection pipeline with optimized hyperparameters.

### 2. Hyperparameter Tuning
```bash
# Fast tuning (for testing)
python hyperparameters/hyperparameter_tuning_fast.py

# Full tuning (comprehensive)
python hyperparameters/hyperparameter_tuning.py
```

### 3. Attack-Specific Evaluation
```bash
python evaluation/evaluate_attack_specific.py
```

### 4. Model Training
```bash
python scripts/train_models_new.py
```

### 5. Comparative Analysis
```bash
# Generate all analysis and visualizations
python analysis/simple_analysis.py

# Generate specific visualizations
python analysis/radar_chart.py
python analysis/performance_tradeoff.py
```

## Key Files

### Core Implementation
- **`anomaly_detection_models.py`** - Main anomaly detection framework with all models
- **`hyperparameters/`** - Complete hyperparameter optimization system

### Data Files
- **`data/features_normal.csv`** - Normal CAN traffic (training data)
- **`data/features_combined.csv`** - Mixed normal + attack (testing data)
- **`data/features_*.csv`** - Attack-specific feature datasets

### Results & Analysis
- **`results/`** - Performance evaluation results
- **`model_comparison_summary.csv`** - Clean performance summary table
- **`comparative_analysis_summary.md`** - Comprehensive model comparison analysis
- **`hyperparameters/hyperparameter_results_analysis.md`** - Hyperparameter tuning analysis

### Visualizations
- **`visualizations/model_comparison_plots.png`** - Performance comparison bar charts
- **`visualizations/model_radar_comparison.png`** - Radar chart of top models
- **`visualizations/performance_tradeoff_analysis.png`** - Speed vs performance trade-offs

## Models Implemented

### Anomaly Detection Models
1. **Local Outlier Factor (LOF)** - Optimized: n_neighbors=20, contamination=0.1
2. **One-Class SVM** - Optimized: nu=0.05, gamma='scale', kernel='rbf'
3. **Elliptic Envelope** - Optimized: support_fraction=0.8, contamination=0.1
4. **Autoencoder** - Optimized: epochs=50, latent_dim=8, dropout=0.0
5. **Isolation Forest** - Optimized: n_estimators=100, max_samples=0.8
6. **DBSCAN** - For comparison (eps=0.5, min_samples=5)

### Performance Rankings (F1-Score)
1. **LOF**: 0.9893 (Best accuracy)
2. **Autoencoder**: 0.9880 (Best for real-time)
3. **One-Class SVM**: 0.9878
4. **Elliptic Envelope**: 0.9870 (Fastest training)
5. **DBSCAN**: 0.3682 (Poor performance)

## 🔧 Configuration

All models use optimized hyperparameters derived from systematic grid search tuning. See `hyperparameters/hyperparameter_results_analysis.md` for detailed analysis.

## 📈 Usage Examples

### Train and Evaluate All Models
```python
from anomaly_detection_models import AnomalyDetectionEvaluator

evaluator = AnomalyDetectionEvaluator()
evaluator.load_data('data/features_normal.csv', 'data/features_combined.csv')

# Train optimized models
evaluator.train_one_class_svm(nu=0.05)
evaluator.train_lof(n_neighbors=20)
evaluator.train_autoencoder(latent_dim=8, epochs=50)

# Evaluate and compare
for model in evaluator.models:
    evaluator.evaluate_model(model)
evaluator.compare_models()
```

### Hyperparameter Tuning
```python
from hyperparameters.hyperparameter_tuning_fast import FastHyperparameterTuner

tuner = FastHyperparameterTuner()
tuner.load_data('data/features_normal.csv', 'data/features_combined.csv')
tuner.tune_lof()
tuner.analyze_results()
```

## Next Steps

1. **Visualization**: Create performance plots and model comparison charts
2. **Real-time Testing**: Evaluate on live CAN bus data
3. **Ensemble Methods**: Combine top-performing models
4. **Production Deployment**: Optimize for real-time inference
