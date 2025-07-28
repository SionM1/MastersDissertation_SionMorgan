CAN Bus Intrusion Detection System (IDS)
Master’s Dissertation – Sion Morgan

This repository contains the implementation, evaluation, and comparative analysis of multiple machine learning models for anomaly detection in Controller Area Network (CAN) bus traffic. Developed as part of a Master’s dissertation in cybersecurity, this work focuses on detecting attacks within automotive systems using unsupervised and semi-supervised approaches.

Overview
The project develops and benchmarks six anomaly detection models on CAN bus datasets representing both normal and malicious behavior. The evaluation includes a comprehensive comparison of detection accuracy, training time, and inference efficiency, with results presented in both tabular and graphical formats.

Implemented Models
Local Outlier Factor (LOF)

One-Class SVM

Elliptic Envelope

Isolation Forest

Autoencoder (Feedforward Neural Network)

DBSCAN (baseline comparison only)

Each model has been tuned using grid search for optimal performance on the selected features.

Summary of Results
LOF achieved the highest accuracy (F1-score: 0.9893)

Autoencoder delivered the fastest inference speed

Elliptic Envelope had the shortest training time

DBSCAN underperformed significantly (F1-score: 0.3682)

A complete performance breakdown is available in comparative_analysis_summary.md.

Visual Output
This repository includes several key visualizations to support model evaluation, including:

Comparative bar charts (F1-Score, AUC, runtime metrics)

Radar charts showing multidimensional performance

Trade-off plots (accuracy vs. runtime performance)

All visualizations are generated programmatically using Python scripts and are located in FeatureExtraction/visualizations.

Repository Structure
css
Copy code
FeatureExtraction/           - Core scripts for model training, testing, and analysis  
├── data/                    - Raw and preprocessed datasets (excluded from repo)  
├── models/                  - Saved models (excluded)  
├── visualizations/          - Performance and comparison plots  
├── simple_analysis.py       - Main analysis pipeline  
├── radar_chart.py           - Radar chart generator  
├── performance_tradeoff.py  - Trade-off visualization  
comparative_analysis_summary.md  
.gitignore  
README.md
Large datasets and model files have been excluded from version control. Please refer to the .gitignore for excluded file types.

Requirements
Python 3.8+
Install dependencies using:

nginx
Copy code
pip install -r requirements.txt
Core libraries used:

scikit-learn

pandas

matplotlib

tensorflow (for Autoencoder)

Running the Project
Place the required dataset files into FeatureExtraction/data/

Run simple_analysis.py to execute the full training and evaluation pipeline

Generate performance visualizations using the provided scripts

Review output metrics and comparative results

Notes
This repository supports academic research conducted at Cardiff University. It is intended for educational and demonstrative purposes. Data used for training and evaluation must be supplied locally and is not included in the repository due to size limitations.

