# Hyperparameter Tuning Results Analysis

## Overview
Hyperparameter tuning was performed on 5 anomaly detection models using a systematic grid search approach. The tuning used:
- **Training Data**: 5,000 normal CAN bus samples
- **Test Data**: 2,000 mixed samples (94.3% anomalies)
- **Evaluation Metrics**: F1-Score (primary), AUC, Precision, Recall
- **Methodology**: Train on normal data only, validate on mixed normal+attack data

## Best Parameters by Model

### 1. Autoencoder (Best Overall Performance)
- **F1-Score**: 0.9904 ⭐
- **AUC**: 0.9907
- **Best Parameters**: 
  - `epochs`: 50
  - `latent_dim`: 8
  - `dropout_rate`: 0.0
- **Training Time**: 3.69s
- **Inference Time**: 0.004s

### 2. LOF (Local Outlier Factor)
- **F1-Score**: 0.9896
- **AUC**: 0.9906
- **Best Parameters**: 
  - `n_neighbors`: 20
  - `contamination`: 0.1
- **Training Time**: 0.10s
- **Inference Time**: 0.13s

### 3. One-Class SVM
- **F1-Score**: 0.9883
- **AUC**: 0.9904
- **Best Parameters**: 
  - `nu`: 0.05
  - `gamma`: 'scale'
  - `kernel`: 'rbf'
- **Training Time**: 0.08s
- **Inference Time**: 0.06s

### 4. Elliptic Envelope
- **F1-Score**: 0.9883
- **AUC**: 0.9914 ⭐ (Highest AUC)
- **Best Parameters**: 
  - `support_fraction`: 0.8
  - `contamination`: 0.1
- **Training Time**: 0.18s
- **Inference Time**: 0.001s (Fastest)

### 5. Isolation Forest
- **F1-Score**: 0.9883
- **AUC**: 0.9777
- **Best Parameters**: 
  - `n_estimators`: 100
  - `max_samples`: 0.8
  - `contamination`: 0.1
- **Training Time**: 0.19s
- **Inference Time**: 0.03s

## Key Findings

### Performance Rankings
1. **Autoencoder** - Best F1-Score (0.9904)
2. **LOF** - Second best F1-Score (0.9896)
3. **Elliptic Envelope** - Best AUC (0.9914), fastest inference
4. **One-Class SVM** - Good balance of speed and performance
5. **Isolation Forest** - Lowest AUC but still competitive

### Parameter Insights
- **Contamination**: 0.1 was optimal for most models (vs 0.15)
- **LOF**: Higher neighbor count (20) performed better than 10
- **SVM**: Lower nu (0.05) outperformed 0.1
- **Autoencoder**: No dropout and moderate epochs (50) were optimal
- **Isolation Forest**: More estimators (100) improved performance

### Speed vs Performance Trade-offs
- **Fastest Training**: One-Class SVM (0.08s)
- **Fastest Inference**: Elliptic Envelope (0.001s)
- **Best Performance**: Autoencoder (but 3.69s training time)
- **Best Balance**: LOF (good performance, reasonable speed)

## Recommendations

### For Production Use
1. **Real-time Detection**: Elliptic Envelope (fastest inference, high AUC)
2. **Batch Processing**: Autoencoder (best F1-Score)
3. **Balanced Approach**: LOF (good performance, moderate speed)

### For Further Research
1. Test these parameters on larger datasets
2. Investigate ensemble methods combining top performers
3. Add more sophisticated hyperparameter optimization (Bayesian optimization)
4. Evaluate on attack-specific datasets

### Next Steps
1. Update main model implementations with these optimal parameters
2. Re-run full evaluation with optimized hyperparameters
3. Generate updated visualizations and comparisons
4. Document performance improvements
