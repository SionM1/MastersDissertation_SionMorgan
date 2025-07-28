#!/usr/bin/env python3
"""
Hyperparameter Tuning for Anomaly Detection Models
Implements grid search tuning using only normal training data
and validates on a fixed small test set of mixed normal+attack data.
"""

import pandas as pd
import numpy as np
import time
import csv
from itertools import product
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    """Simple Autoencoder for anomaly detection with dropout"""
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16, dropout_rate=0.1):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse.numpy()

class HyperparameterTuner:
    """Hyperparameter tuning for anomaly detection models"""
    
    def __init__(self, results_file='hyperparameter_results.csv'):
        self.scaler = RobustScaler()
        self.results_file = results_file
        self.results = []
        
        # Initialize CSV file
        self._init_results_file()
    
    def _init_results_file(self):
        """Initialize the results CSV file with headers"""
        headers = [
            'model', 'fold', 'timestamp', 'precision', 'recall', 'f1_score', 'auc',
            'training_time', 'inference_time'
        ]
        
        # Add parameter headers (will be filled dynamically)
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers + ['parameters'])
    
    def load_data(self, normal_path, test_path, train_subset=20000, test_subset=10000):
        """Load and prepare data for hyperparameter tuning"""
        print("Loading data for hyperparameter tuning...")
        
        # Load normal training data
        normal_data = pd.read_csv(normal_path)
        
        # Load test data (mixed normal+attack)
        test_data = pd.read_csv(test_path)
        
        # Use smaller subsets for efficient tuning
        if len(normal_data) > train_subset:
            normal_data = normal_data.sample(n=train_subset, random_state=42)
        if len(test_data) > test_subset:
            test_data = test_data.sample(n=test_subset, random_state=42)
        
        print(f"   Training set: {len(normal_data):,} samples")
        print(f"   Test set: {len(test_data):,} samples")
        
        # Prepare features
        feature_cols = [col for col in normal_data.columns if col != 'label']
        
        self.X_train = normal_data[feature_cols].values
        self.X_test = test_data[feature_cols].values
        self.y_test = (test_data['label'] != 'normal').astype(int)
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Test anomalies: {self.y_test.sum()} ({self.y_test.mean()*100:.1f}%)")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def _log_result(self, model_name, fold, params, metrics, training_time, inference_time):
        """Log result to CSV file"""
        row = [
            model_name, fold, time.strftime('%Y-%m-%d %H:%M:%S'),
            metrics['precision'], metrics['recall'], metrics['f1'], metrics['auc'],
            training_time, inference_time, str(params)
        ]
        
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _evaluate_predictions(self, y_true, y_pred, anomaly_scores=None):
        """Calculate evaluation metrics"""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, anomaly_scores) if anomaly_scores is not None else None
        except:
            auc = None
            
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def tune_lof(self, n_neighbors_range=[10, 20, 30], contamination_range=[0.05, 0.1, 0.15]):
        """Tune LOF hyperparameters"""
        print("\nTuning LOF hyperparameters...")
        
        param_combinations = list(product(n_neighbors_range, contamination_range))
        
        for i, (n_neighbors, contamination) in enumerate(param_combinations):
            print(f"   Testing LOF {i+1}/{len(param_combinations)}: neighbors={n_neighbors}, contamination={contamination}")
            
            # Train model
            start_time = time.time()
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
            model.fit(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            predictions = model.predict(self.X_test_scaled)
            y_pred = (predictions == -1).astype(int)
            
            try:
                anomaly_scores = -model.decision_function(self.X_test_scaled)
            except:
                anomaly_scores = None
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._evaluate_predictions(self.y_test, y_pred, anomaly_scores)
            
            # Log results
            params = {'n_neighbors': n_neighbors, 'contamination': contamination}
            self._log_result('LOF', i, params, metrics, training_time, inference_time)
            
            auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
            print(f"      F1: {metrics['f1']:.4f}, AUC: {auc_str}")
    
    def tune_one_class_svm(self, nu_range=[0.01, 0.05, 0.1], gamma_range=['scale', 'auto'], kernel_range=['rbf']):
        """Tune One-Class SVM hyperparameters"""
        print("\nTuning One-Class SVM hyperparameters...")
        
        param_combinations = list(product(nu_range, gamma_range, kernel_range))
        
        for i, (nu, gamma, kernel) in enumerate(param_combinations):
            print(f"   Testing SVM {i+1}/{len(param_combinations)}: nu={nu}, gamma={gamma}, kernel={kernel}")
            
            # Train model
            start_time = time.time()
            model = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)
            model.fit(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            predictions = model.predict(self.X_test_scaled)
            y_pred = (predictions == -1).astype(int)
            
            try:
                anomaly_scores = -model.decision_function(self.X_test_scaled)
            except:
                anomaly_scores = None
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._evaluate_predictions(self.y_test, y_pred, anomaly_scores)
            
            # Log results
            params = {'nu': nu, 'gamma': gamma, 'kernel': kernel}
            self._log_result('OneClassSVM', i, params, metrics, training_time, inference_time)
            
            auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
            print(f"      F1: {metrics['f1']:.4f}, AUC: {auc_str}")
    
    def tune_elliptic_envelope(self, support_fraction_range=[None, 0.5, 0.8], contamination_range=[0.05, 0.1, 0.15]):
        """Tune Elliptic Envelope hyperparameters"""
        print("\nTuning Elliptic Envelope hyperparameters...")
        
        param_combinations = list(product(support_fraction_range, contamination_range))
        
        for i, (support_fraction, contamination) in enumerate(param_combinations):
            print(f"   Testing Elliptic {i+1}/{len(param_combinations)}: support_fraction={support_fraction}, contamination={contamination}")
            
            # Train model
            start_time = time.time()
            model = EllipticEnvelope(support_fraction=support_fraction, contamination=contamination)
            model.fit(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            predictions = model.predict(self.X_test_scaled)
            y_pred = (predictions == -1).astype(int)
            
            try:
                anomaly_scores = -model.decision_function(self.X_test_scaled)
            except:
                anomaly_scores = None
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._evaluate_predictions(self.y_test, y_pred, anomaly_scores)
            
            # Log results
            params = {'support_fraction': support_fraction, 'contamination': contamination}
            self._log_result('EllipticEnvelope', i, params, metrics, training_time, inference_time)
            
            auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
            print(f"      F1: {metrics['f1']:.4f}, AUC: {auc_str}")
    
    def tune_isolation_forest(self, n_estimators_range=[50, 100, 200], max_samples_range=[0.5, 0.8, 1.0], contamination_range=[0.05, 0.1, 0.15]):
        """Tune Isolation Forest hyperparameters"""
        print("\nTuning Isolation Forest hyperparameters...")
        
        param_combinations = list(product(n_estimators_range, max_samples_range, contamination_range))
        
        for i, (n_estimators, max_samples, contamination) in enumerate(param_combinations):
            print(f"   Testing IsolationForest {i+1}/{len(param_combinations)}: n_estimators={n_estimators}, max_samples={max_samples}, contamination={contamination}")
            
            # Train model
            start_time = time.time()
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=42)
            model.fit(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            predictions = model.predict(self.X_test_scaled)
            y_pred = (predictions == -1).astype(int)
            
            try:
                anomaly_scores = -model.decision_function(self.X_test_scaled)
            except:
                anomaly_scores = None
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._evaluate_predictions(self.y_test, y_pred, anomaly_scores)
            
            # Log results
            params = {'n_estimators': n_estimators, 'max_samples': max_samples, 'contamination': contamination}
            self._log_result('IsolationForest', i, params, metrics, training_time, inference_time)
            
            auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
            print(f"      F1: {metrics['f1']:.4f}, AUC: {auc_str}")
    
    def tune_autoencoder(self, epochs_range=[50, 100], latent_dim_range=[8, 16, 32], dropout_range=[0.0, 0.1, 0.2]):
        """Tune Autoencoder hyperparameters"""
        print("\nTuning Autoencoder hyperparameters...")
        
        param_combinations = list(product(epochs_range, latent_dim_range, dropout_range))
        
        input_dim = self.X_train_scaled.shape[1]
        
        for i, (epochs, latent_dim, dropout_rate) in enumerate(param_combinations):
            print(f"   Testing Autoencoder {i+1}/{len(param_combinations)}: epochs={epochs}, latent_dim={latent_dim}, dropout={dropout_rate}")
            
            # Train model
            start_time = time.time()
            model = Autoencoder(input_dim, hidden_dim=64, latent_dim=latent_dim, dropout_rate=dropout_rate)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Prepare data
            train_tensor = torch.FloatTensor(self.X_train_scaled)
            train_dataset = TensorDataset(train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                for batch_idx, (data,) in enumerate(train_loader):
                    optimizer.zero_grad()
                    reconstructed = model(data)
                    loss = criterion(reconstructed, data)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            training_time = time.time() - start_time
            
            # Evaluate
            start_time = time.time()
            test_tensor = torch.FloatTensor(self.X_test_scaled)
            reconstruction_errors = model.get_reconstruction_error(test_tensor)
            
            # Calculate threshold from training data
            train_tensor = torch.FloatTensor(self.X_train_scaled)
            train_errors = model.get_reconstruction_error(train_tensor)
            threshold = np.percentile(train_errors, 95)
            
            y_pred = (reconstruction_errors > threshold).astype(int)
            anomaly_scores = reconstruction_errors
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._evaluate_predictions(self.y_test, y_pred, anomaly_scores)
            
            # Log results
            params = {'epochs': epochs, 'latent_dim': latent_dim, 'dropout_rate': dropout_rate}
            self._log_result('Autoencoder', i, params, metrics, training_time, inference_time)
            
            auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
            print(f"      F1: {metrics['f1']:.4f}, AUC: {auc_str}")
    
    def tune_dbscan(self, eps_range=[0.3, 0.5, 0.7], min_samples_range=[3, 5, 10]):
        """Tune DBSCAN hyperparameters (simplified approach)"""
        print("\nTuning DBSCAN hyperparameters...")
        
        param_combinations = list(product(eps_range, min_samples_range))
        
        for i, (eps, min_samples) in enumerate(param_combinations):
            print(f"   Testing DBSCAN {i+1}/{len(param_combinations)}: eps={eps}, min_samples={min_samples}")
            
            # Train model
            start_time = time.time()
            model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = model.fit_predict(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # Calculate outlier rate from training
            outlier_rate = np.mean(cluster_labels == -1)
            
            # Evaluate (simplified: random sampling based on outlier rate)
            start_time = time.time()
            np.random.seed(42)
            n_outliers = int(len(self.X_test_scaled) * outlier_rate)
            y_pred = np.zeros(len(self.X_test_scaled))
            if n_outliers > 0:
                outlier_indices = np.random.choice(len(self.X_test_scaled), n_outliers, replace=False)
                y_pred[outlier_indices] = 1
            y_pred = y_pred.astype(int)
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._evaluate_predictions(self.y_test, y_pred, y_pred.astype(float))
            
            # Log results
            params = {'eps': eps, 'min_samples': min_samples, 'outlier_rate': outlier_rate}
            self._log_result('DBSCAN', i, params, metrics, training_time, inference_time)
            
            print(f"      F1: {metrics['f1']:.4f}, Outlier Rate: {outlier_rate:.3f}")
    
    def analyze_results(self):
        """Analyze and summarize tuning results"""
        print(f"\nAnalyzing results from {self.results_file}...")
        
        # Read results
        results_df = pd.read_csv(self.results_file)
        
        if len(results_df) == 0:
            print("No results found!")
            return
        
        print("\nBest parameters per model (by F1-score):")
        print("="*60)
        
        summary_results = []
        
        for model in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model]
            best_idx = model_results['f1_score'].idxmax()
            best_result = model_results.loc[best_idx]
            
            summary_results.append({
                'Model': model,
                'Best_F1': best_result['f1_score'],
                'Best_AUC': best_result['auc'],
                'Best_Precision': best_result['precision'],
                'Best_Recall': best_result['recall'],
                'Best_Parameters': best_result['parameters'],
                'Training_Time': best_result['training_time'],
                'Inference_Time': best_result['inference_time']
            })
            
            print(f"\n{model}:")
            print(f"   F1-Score: {best_result['f1_score']:.4f}")
            print(f"   AUC: {best_result['auc']:.4f if pd.notna(best_result['auc']) else 'N/A'}")
            print(f"   Parameters: {best_result['parameters']}")
        
        # Save summary
        summary_df = pd.DataFrame(summary_results)
        summary_file = 'hyperparameter_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
        return summary_df

def main():
    """Main execution function"""
    print("Hyperparameter Tuning for Anomaly Detection Models")
    print("="*60)
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Load data
    tuner.load_data(
        normal_path='../data/features_normal.csv',
        test_path='../data/features_combined.csv'
    )
    
    # Tune each model
    tuner.tune_lof()
    tuner.tune_one_class_svm()
    tuner.tune_elliptic_envelope()
    tuner.tune_isolation_forest()
    tuner.tune_autoencoder()
    tuner.tune_dbscan()  # Low priority
    
    # Analyze results
    tuner.analyze_results()
    
    print("\nHyperparameter tuning completed!")
    print("Check hyperparameter_results.csv for detailed results")
    print("Check hyperparameter_summary.csv for best parameters per model")

if __name__ == "__main__":
    main()
