import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
from tqdm import tqdm
import time
import joblib
warnings.filterwarnings('ignore')

def main():
    print("Loading full dataset...")
    df = pd.read_csv('features_combined.csv')
    print(f"Original shape: {df.shape}")
    print(f"Original labels:\n{df['label'].value_counts()}")
    
    # SAMPLE DATA TO MANAGEABLE SIZE
    print("\nSampling 100,000 records...")
    with tqdm(total=3, desc="Sampling Data", ncols=80) as pbar:
        normal_df = df[df['label'] == 'normal'].sample(n=50000, random_state=42)
        pbar.update(1)
        
        attack_df = df[df['label'] == 'attack'].sample(n=50000, random_state=42)
        pbar.update(1)
        
        # Combine and shuffle
        df_sample = pd.concat([normal_df, attack_df], ignore_index=True)
        df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        pbar.update(1)
    
    print(f"Sampled shape: {df_sample.shape}")
    print(f"Sampled labels:\n{df_sample['label'].value_counts()}")
    
    # Prepare features
    print("\nPreparing features and labels...")
    with tqdm(total=2, desc="Data Preparation", ncols=80) as pbar:
        X = df_sample.drop('label', axis=1)
        y = df_sample['label']
        pbar.update(1)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        pbar.update(1)
    
    print(f"Label encoding: normal=1, attack=0")
    
    # Split data
    print("\nSplitting and scaling data...")
    with tqdm(total=3, desc="Data Split & Scale", ncols=80) as pbar:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        pbar.update(1)
        
        # Scale data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        pbar.update(1)
        
        X_test_scaled = scaler.transform(X_test)
        pbar.update(1)
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    results = {}
    
    # 1. RANDOM FOREST (unscaled)
    print("\n=== RANDOM FOREST ===")
    with tqdm(total=3, desc="Random Forest", ncols=80) as pbar:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
        pbar.set_description("RF: Training")
        rf.fit(X_train, y_train)
        pbar.update(1)
        
        pbar.set_description("RF: Predicting")
        rf_pred = rf.predict(X_test)
        pbar.update(1)
        
        rf_proba = rf.predict_proba(X_test)[:, 1]
        pbar.update(1)
        pbar.set_description("RF: Complete")
    
    results['RF'] = {
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'F1': f1_score(y_test, rf_pred),
        'AUC': roc_auc_score(y_test, rf_proba)
    }
    print(f"RF Results: P={results['RF']['Precision']:.3f}, R={results['RF']['Recall']:.3f}, F1={results['RF']['F1']:.3f}, AUC={results['RF']['AUC']:.3f}")
    
    # 2. SVM (scaled)
    print("\n=== SVM ===")
    with tqdm(total=4, desc="SVM", ncols=80) as pbar:
        # Use smaller subset for SVM training
        pbar.set_description("SVM: Sampling 20K subset")
        svm_sample_size = 20000
        sample_indices = np.random.choice(len(X_train_scaled), svm_sample_size, replace=False)
        X_svm_train = X_train_scaled[sample_indices]
        y_svm_train = y_train[sample_indices]
        pbar.update(1)
        
        # Use linear kernel (much faster)
        pbar.set_description("SVM: Training (linear)")
        svm = SVC(kernel='linear', probability=True, random_state=42, C=1.0, max_iter=1000)
        svm.fit(X_svm_train, y_svm_train)
        pbar.update(1)
        
        pbar.set_description("SVM: Predicting")
        svm_pred = svm.predict(X_test_scaled)
        pbar.update(1)
        
        svm_proba = svm.predict_proba(X_test_scaled)[:, 1]
        pbar.update(1)
        pbar.set_description("SVM: Complete")
    
    results['SVM'] = {
        'Precision': precision_score(y_test, svm_pred),
        'Recall': recall_score(y_test, svm_pred),
        'F1': f1_score(y_test, svm_pred),
        'AUC': roc_auc_score(y_test, svm_proba)
    }
    print(f"SVM Results: P={results['SVM']['Precision']:.3f}, R={results['SVM']['Recall']:.3f}, F1={results['SVM']['F1']:.3f}, AUC={results['SVM']['AUC']:.3f}")
    
    # 3. ISOLATION FOREST (unscaled)
    print("\n=== ISOLATION FOREST ===")
    with tqdm(total=4, desc="Isolation Forest", ncols=80) as pbar:
        # Train only on normal data
        X_normal = X_train[y_train == 1]  # normal=1
        pbar.set_description(f"ISO: Training on {len(X_normal)} normal samples")
        pbar.update(1)
        
        iso = IsolationForest(n_estimators=100, max_samples=0.8, contamination=0.1, random_state=42, n_jobs=1)
        iso.fit(X_normal)
        pbar.update(1)
        
        pbar.set_description("ISO: Predicting")
        iso_pred_raw = iso.predict(X_test)
        iso_pred = np.where(iso_pred_raw == 1, 1, 0)  # 1=normal, -1=attack->0
        pbar.update(1)
        
        iso_scores = iso.decision_function(X_test)
        iso_proba = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        pbar.update(1)
        pbar.set_description("ISO: Complete")
    
    results['ISO'] = {
        'Precision': precision_score(y_test, iso_pred),
        'Recall': recall_score(y_test, iso_pred),
        'F1': f1_score(y_test, iso_pred),
        'AUC': roc_auc_score(y_test, iso_proba)
    }
    print(f"ISO Results: P={results['ISO']['Precision']:.3f}, R={results['ISO']['Recall']:.3f}, F1={results['ISO']['F1']:.3f}, AUC={results['ISO']['AUC']:.3f}")
    
    # DISPLAY RESULTS
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    print("\nSaving results...")
    results_df.to_csv('results.csv')
    print("Results saved to results.csv")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Best Precision: {results_df['Precision'].idxmax()} ({results_df['Precision'].max():.4f})")
    print(f"Best Recall: {results_df['Recall'].idxmax()} ({results_df['Recall'].max():.4f})")
    print(f"Best F1-Score: {results_df['F1'].idxmax()} ({results_df['F1'].max():.4f})")
    print(f"Best AUC: {results_df['AUC'].idxmax()} ({results_df['AUC'].max():.4f})")
    
    # ADD THIS SECTION - SAVE TRAINED MODELS
    print("\n" + "="*50)
    print("SAVING TRAINED MODELS AND PREPROCESSORS")
    print("="*50)
    
    # Save models
    print("Saving Random Forest model...")
    joblib.dump(rf, 'random_forest_model.pkl')
    
    print("Saving Isolation Forest model...")
    joblib.dump(iso, 'isolation_forest_model.pkl')
    
    # Save preprocessors
    print("Saving RobustScaler...")
    joblib.dump(scaler, 'robust_scaler.pkl')
    
    print("Saving LabelEncoder...")
    joblib.dump(le, 'label_encoder.pkl')
    
    print("\nAll models and preprocessors saved successfully!")
    print("Files created:")
    print("- random_forest_model.pkl (Best performer)")
    print("- isolation_forest_model.pkl") 
    print("- robust_scaler.pkl (For preprocessing new data)")
    print("- label_encoder.pkl (For label conversion)")
    print("- results.csv (Performance metrics)")

if __name__ == "__main__":
    main()
