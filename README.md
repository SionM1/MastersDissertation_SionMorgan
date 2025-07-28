# CAN Bus Intrusion Detection System

A comprehensive machine learning-based intrusion detection system for Controller Area Network (CAN) bus security, implementing multiple anomaly detection algorithms with hyperparameter optimization and comparative analysis.

## Project Overview

This project implements and evaluates various machine learning models for detecting intrusions in CAN bus networks. The system uses unsupervised anomaly detection techniques to identify malicious activities in automotive network traffic.

## Key Features

- **Multiple Anomaly Detection Models**: LOF, One-Class SVM, Autoencoder, Elliptic Envelope, Isolation Forest, DBSCAN
- **Hyperparameter Optimization**: Systematic grid search tuning for all models
- **Comprehensive Evaluation**: Attack-specific analysis across different attack types (DoS, Fuzzy, Gear, RPM)
- **Performance Visualization**: Comparative charts, radar plots, and trade-off analysis
- **Real-time Capability**: Optimized models for real-time intrusion detection

## Project Structure

```
CAN-IDS/
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
├── AttackData/                            # Raw attack datasets
├── StandardizedData/                      # Processed and standardized datasets
├── DatasetCVSConversion/                  # Data conversion utilities
├── FeatureExtraction/                     # Main analysis framework
│   ├── README.md                          # Detailed feature extraction documentation
│   ├── anomaly_detection_models.py       # Core anomaly detection framework
│   ├── data/                              # Feature datasets
│   ├── models/                            # Trained ML models
│   ├── results/                           # Evaluation results
│   ├── analysis/                          # Comparative analysis scripts
│   ├── visualizations/                    # Generated plots and charts
│   ├── evaluation/                        # Attack-specific evaluation
│   ├── scripts/                           # Utility scripts
│   └── hyperparameters/                   # Hyperparameter tuning
└── convert_rpm_to_standard.py            # Data standardization script
```

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <your-new-repo-url>
cd CAN-IDS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main anomaly detection framework:
```bash
cd FeatureExtraction
python anomaly_detection_models.py
```

### Generate Analysis and Visualizations

```bash
cd FeatureExtraction/analysis
python simple_analysis.py
```

## Model Performance Results

### Best Performing Models (F1-Score)

1. **LOF (Local Outlier Factor)**: 0.9893
   - Best overall accuracy
   - Optimized parameters: n_neighbors=20, contamination=0.1

2. **Autoencoder**: 0.9880
   - Best for real-time detection (fast inference: 0.040s)
   - Optimized parameters: epochs=50, latent_dim=8, dropout=0.0

3. **One-Class SVM**: 0.9878
   - Balanced performance
   - Optimized parameters: nu=0.05, gamma='scale', kernel='rbf'

4. **Elliptic Envelope**: 0.9870
   - Fastest training (1.201s)
   - Optimized parameters: support_fraction=0.8, contamination=0.1

### Performance Characteristics

- **Highest Precision**: Autoencoder (0.9969)
- **Highest Recall**: Elliptic Envelope (0.9801)
- **Fastest Training**: Elliptic Envelope (1.201s)
- **Fastest Inference**: Autoencoder (0.040s)
- **Best AUC**: LOF (0.9926)

## Key Components

### 1. Feature Extraction (`FeatureExtraction/`)
- Complete anomaly detection framework
- Hyperparameter optimization system
- Comparative analysis and visualization tools

### 2. Data Processing
- Raw CAN bus data preprocessing
- Feature engineering and extraction
- Attack-specific dataset creation

### 3. Model Evaluation
- Cross-validation and performance metrics
- Attack-specific evaluation (DoS, Fuzzy, Gear, RPM)
- Speed vs performance trade-off analysis

## Research Contributions

1. **Comprehensive Model Comparison**: Systematic evaluation of 6 different anomaly detection algorithms on CAN bus data

2. **Hyperparameter Optimization**: Grid search optimization for all models with detailed analysis

3. **Attack-Specific Analysis**: Performance evaluation across different types of CAN bus attacks

4. **Real-time Performance Analysis**: Speed vs accuracy trade-off evaluation for production deployment

5. **Visualization Framework**: Comprehensive plotting and analysis tools for model comparison

## Usage Examples

### Train All Models with Optimized Parameters
```python
from FeatureExtraction.anomaly_detection_models import AnomalyDetectionEvaluator

evaluator = AnomalyDetectionEvaluator()
evaluator.load_data('data/features_normal.csv', 'data/features_combined.csv')

# Train optimized models
evaluator.train_lof(n_neighbors=20, contamination=0.1)
evaluator.train_autoencoder(latent_dim=8, epochs=50)
evaluator.train_one_class_svm(nu=0.05)

# Evaluate and compare
for model in evaluator.models:
    evaluator.evaluate_model(model)
evaluator.compare_models()
```

### Generate Comparative Analysis
```python
from FeatureExtraction.analysis.simple_analysis import main
main()  # Generates summary CSV and comparison plots
```

## Deployment Recommendations

### Real-time CAN Bus Monitoring
- **Recommended Model**: Autoencoder
- **Rationale**: High accuracy (F1: 0.9880) with fast inference (0.040s)

### Resource-Constrained Environments  
- **Recommended Model**: Elliptic Envelope
- **Rationale**: Fastest training (1.201s) with good accuracy (F1: 0.9870)

### Maximum Accuracy Applications
- **Recommended Model**: LOF
- **Rationale**: Highest F1-Score (0.9893) and AUC (0.9926)

## File Descriptions

### Core Files
- `FeatureExtraction/anomaly_detection_models.py` - Main framework with all models
- `FeatureExtraction/results/model_comparison_summary.csv` - Performance summary
- `FeatureExtraction/analysis/comparative_analysis_summary.md` - Detailed analysis

### Visualizations
- `FeatureExtraction/visualizations/model_comparison_plots.png` - Performance bar charts
- `FeatureExtraction/visualizations/model_radar_comparison.png` - Multi-metric radar chart
- `FeatureExtraction/visualizations/performance_tradeoff_analysis.png` - Speed vs accuracy plots

### Data Files
- `FeatureExtraction/data/features_normal.csv` - Normal CAN traffic features
- `FeatureExtraction/data/features_combined.csv` - Mixed normal + attack features
- `FeatureExtraction/data/features_*.csv` - Attack-specific datasets

## Technical Details

### Methodology
- **Training Data**: Normal CAN bus traffic only (unsupervised learning)
- **Testing Data**: Mixed normal and attack traffic
- **Evaluation Metrics**: F1-Score, AUC, Precision, Recall, Training/Inference Time
- **Cross-Validation**: Proper anomaly detection evaluation methodology

### Hyperparameter Optimization
- **Method**: Grid search with 2-3 values per parameter
- **Validation**: Fixed test set of mixed normal+attack data
- **Results**: Documented in `FeatureExtraction/hyperparameters/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of academic research. Please cite appropriately if used in academic work.

## Citation

If you use this work in your research, please cite:

```
@thesis{YourName2025,
  title={CAN Bus Intrusion Detection Using Machine Learning: A Comparative Analysis},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## Contact

For questions or collaborations, please contact [your-email@university.edu]

## Acknowledgments

- Dataset providers and CAN bus security research community
- Open source machine learning libraries (scikit-learn, PyTorch, pandas, matplotlib)
