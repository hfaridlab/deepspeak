import os
import glob
import joblib
import pandas as pd
import numpy as np
import yaml
import re
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix

def load_config(config_path):
    """ Load configuration from a YAML file. """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def load_embeddings(embedding_file):
    """Ingest embeddings from CSV file"""
    if isinstance(embedding_file, list) and len(embedding_file) > 1:
        dataframes = [pd.read_csv(file) for file in embedding_file]
        return pd.concat(dataframes, ignore_index=True)
    return pd.read_csv(embedding_file[0])

def prepare_data(df):
    """Prepare input data for each embedding type for testing"""
    if 'label' in df.columns:
        df.drop(columns=['label'], inplace=True)
    df['label'] = df['audio_generator'].apply(lambda x: 1 if x == 'real' else 0)
    X = df.drop(columns=['file_path', 'audio_generator', 'is_train', 'label'])
    y = df['label']
    return X, y

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    f_score = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    real_acc = tp / (tp + fn)
    fake_acc = tn / (tn + fp)

    return {
        'Acc': accuracy,
        'AUC': auc,
        'EER': eer,
        'F': f_score,
        'real_acc': real_acc,
        'fake_acc': fake_acc
    }

def main():
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['classifiers_cross_testing']
    
    # Set up directories
    model_dir = paths['model_dir']
    embedding_base_dir = paths['embedding_base_dir']
    results_dir = paths['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Output file
    results_csv_path = os.path.join(results_dir, paths['results_csv'])
    
    # Get all model files
    model_files = glob.glob(os.path.join(model_dir, '*.joblib'))
    
    # Define datasets
    datasets = ['asvspoof', 'TIMIT_ElevenLabs', 'deepspeak']
    
    # Initialize results
    results = []
    
    # Process each model
    for model_file in model_files:
        filename = os.path.basename(model_file)
        match = re.match(r'(.+?)_(.+?)_(.+?)\.joblib', filename)
        if not match:
            continue
            
        embedding_type, trained_on, model_type = match.groups()
        model_type = 'lr' if 'Logistic_Regression' in model_type else 'rf'
        
        # Load the model
        model = joblib.load(model_file)
        
        row = {
            'Embedding_type': embedding_type,
            'Classifier': model_type,
            'Trained_On': trained_on
        }
        
        # Test on each dataset
        for dataset in datasets:
            embedding_file = glob.glob(os.path.join(embedding_base_dir, dataset, f'*_{embedding_type}_*.csv'))
            if not embedding_file:
                for metric in ['Acc', 'AUC', 'EER', 'F', 'real_acc', 'fake_acc']:
                    row[f'{dataset}_{metric}'] = np.nan
                continue
                
            # Load and prepare data
            df = load_embeddings(embedding_file)
            X, y = prepare_data(df)
            
            # Use test set if available, otherwise use all data
            if 'is_train' in df.columns:
                X_test = X[df['is_train'] == 0]
                y_test = y[df['is_train'] == 0]
            else:
                X_test = X
                y_test = y
                
            # Make predictions
            try:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                
                for metric, value in metrics.items():
                    row[f'{dataset}_{metric}'] = value
                    
            except Exception as e:
                for metric in ['Acc', 'AUC', 'EER', 'F', 'real_acc', 'fake_acc']:
                    row[f'{dataset}_{metric}'] = np.nan
        
        results.append(row)
    
    # Convert to DataFrame and reorder columns
    results_df = pd.DataFrame(results)
    
    column_order = ['Embedding_type', 'Classifier', 'Trained_On']
    for dataset in datasets:
        for metric in ['Acc', 'AUC', 'EER', 'F', 'real_acc', 'fake_acc']:
            column_order.append(f'{dataset}_{metric}')
    
    for col in column_order:
        if col not in results_df.columns:
            results_df[col] = np.nan
    
    results_df = results_df[column_order]
    
    # Save results
    results_df.to_csv(results_csv_path, index=False)

if __name__ == "__main__":
    main()