import os
import glob
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.utils import resample
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary or None if loading fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def load_embeddings(embedding_files):
    """Ingest embeddings from CSV file"""

    v1_1_files = [f for f in embedding_files if 'v1_1' in f]
    v2_files = [f for f in embedding_files if 'v2' in f]

    v1_1_data = [pd.read_csv(f) for f in v1_1_files]
    v2_data = [pd.read_csv(f) for f in v2_files]
    
    return pd.concat(v1_1_data, ignore_index=True), pd.concat(v2_data, ignore_index=True)

def prepare_data(df):
    """Prepare input data for each embedding type for training and testing"""

    # Original label col was for video - drop and overwrite
    df.drop(columns=['label'], inplace=True)
    df['label'] = df['audio_generator'].apply(lambda x: 1 if x == 'real' else 0)
    X = df.drop(columns=['file_path', 'audio_generator', 'is_train', 'label'])
    y = df['label']

    return X, y

def balance_data(X, y, random_state=42):
    """Balance the dataset by downsampling the larger class"""
    # Separate majority and minority classes
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]
    
    # Downsample majority class
    if len(X_majority) > len(X_minority):
        X_majority_downsampled, y_majority_downsampled = resample(
            X_majority, y_majority,
            replace=False,
            n_samples=len(X_minority),
            random_state=random_state
        )
        X_balanced = pd.concat([X_majority_downsampled, X_minority])
        y_balanced = pd.concat([y_majority_downsampled, y_minority])
    else:
        X_minority_downsampled, y_minority_downsampled = resample(
            X_minority, y_minority,
            replace=False,
            n_samples=len(X_majority),
            random_state=random_state
        )
        X_balanced = pd.concat([X_majority, X_minority_downsampled])
        y_balanced = pd.concat([y_majority, y_minority_downsampled])
    
    return X_balanced, y_balanced

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate evaluation metrics:
    ACCURACY, AUC, EER, F-SCORE, PRECISION, RECALL, CLASS ACCURACY (REAL), CLASS ACCURACY (FAKE), FALSE POSITIVES, FALSE NEGATIVES"""

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    f_score = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    class_accuracies = [tn / (tn + fp), tp / (tp + fn)]
    
    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'EER': eer,
        'F-score': f_score,
        'Precision': precision,
        'Recall': recall,
        'Class Accuracy Real': class_accuracies[1],
        'Class Accuracy Fake': class_accuracies[0],
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp
    }

def train_and_evaluate(X_train, y_train, X_test, y_test, model, model_name, embedding_type, embedding_name, n_folds=5):
    """Train and evaluate model performance"""

    print(f"\nStarting training for {model_name} on {embedding_type} embeddings")

    # Balance the training data
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    print(f"Balanced training data: {X_train_balanced.shape} {y_train_balanced.shape}")

    # Train the model on the entire balanced training set
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on the test set
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics using test data
    metrics = calculate_metrics(y_test, y_pred_test, y_prob_test)
    metrics['Model'] = model_name
    metrics['Embedding Type'] = embedding_type
    metrics['Embedding Name'] = embedding_name

    print(f"Completed evaluation for {model_name} on {embedding_type} + {embedding_name}embeddings")
    print(f"Metrics: {metrics}\n")

    return metrics, model

def main():
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['classifiers']['deepspeak']
    
    dataset = 'deepspeak'
    embedding_dir = paths['embedding_dir']
    results_dir = paths['results_dir']
    model_output_dir = paths['model_output_dir']
    results_csv_path = os.path.join(results_dir, f'{dataset}_embeddings_train_test_model_results.csv')
    
    os.makedirs(results_dir, exist_ok=True)
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    all_results = []

    embedding_types = list(set(file.split('_')[-2] for file in os.listdir(embedding_dir) if file.endswith('.csv')))

    for embedding in embedding_types:

        # Ingest all embeddings files
        embedding_files = glob.glob(os.path.join(embedding_dir, f'*_{embedding}_*.csv'))
        print(f'Loading {embedding} embeddings')
        v1_1_data, v2_data = load_embeddings(embedding_files)
        
        # Prepare input data
        print(f'{embedding}: Ingesting v1_1 and v2 data')
        X_v1_1, y_v1_1 = prepare_data(v1_1_data)
        X_v2, y_v2 = prepare_data(v2_data)
        X_combined = pd.concat([X_v1_1, X_v2], ignore_index=True)
        y_combined = pd.concat([y_v1_1, y_v2], ignore_index=True)
        
        # Split into train and test sets based on 'is_train' column
        X_v1_1_train, y_v1_1_train = X_v1_1[v1_1_data['is_train'] == 1], y_v1_1[v1_1_data['is_train'] == 1]
        X_v1_1_test, y_v1_1_test = X_v1_1[v1_1_data['is_train'] == 0], y_v1_1[v1_1_data['is_train'] == 0]
        X_v2_train, y_v2_train = X_v2[v2_data['is_train'] == 1], y_v2[v2_data['is_train'] == 1]
        X_v2_test, y_v2_test = X_v2[v2_data['is_train'] == 0], y_v2[v2_data['is_train'] == 0]
        X_combined_train, y_combined_train = X_combined[y_combined.index.isin(v1_1_data[v1_1_data['is_train'] == 1].index) | y_combined.index.isin(v2_data[v2_data['is_train'] == 1].index)], y_combined[y_combined.index.isin(v1_1_data[v1_1_data['is_train'] == 1].index) | y_combined.index.isin(v2_data[v2_data['is_train'] == 1].index)]
        X_combined_test, y_combined_test = X_combined[y_combined.index.isin(v1_1_data[v1_1_data['is_train'] == 0].index) | y_combined.index.isin(v2_data[v2_data['is_train'] == 0].index)], y_combined[y_combined.index.isin(v1_1_data[v1_1_data['is_train'] == 0].index) | y_combined.index.isin(v2_data[v2_data['is_train'] == 0].index)]
        
        print(f'{embedding} v1_1 data:')
        print(X_v1_1.head())
        print(f'{embedding} v2_data:')
        print(X_v2.head())
        
        # Define models - here we use one linear (logistic regression) and one non-linear (random forest)
        models = [
            (LogisticRegression(max_iter=1000), 'Logistic Regression'),
            (RandomForestClassifier(n_estimators=100), 'Random Forest')
        ]
        
        # Generate results
        results = []
        print(f'{embedding}: Training and evaluating models')
        for model, model_name in models:
            for (X_train, y_train, X_test, y_test, embedding_type) in [
            (X_v1_1_train, y_v1_1_train, X_v1_1_test, y_v1_1_test, 'v1_1'),
            (X_v2_train, y_v2_train, X_v2_test, y_v2_test, 'v2'),
            (X_combined_train, y_combined_train, X_combined_test, y_combined_test, 'combined')]:
                
                metrics, model = train_and_evaluate(X_train, y_train, X_test, y_test, model, model_name, embedding_type, embedding)
        
                # Convert metrics to DataFrame
                results_df = pd.DataFrame([metrics])
                # Reorder columns so 'Embedding Name', 'Embedding Type', and 'Model' are the first three columns
                column_order = ['Embedding Name', 'Model', 'Embedding Type'] + [col for col in results_df.columns if col not in ['Embedding Name', 'Embedding Type', 'Model']]
                results_df = results_df[column_order]
                
                # Append results to the CSV file
                results_df.to_csv(results_csv_path, mode='a', header=not os.path.exists(results_csv_path), index=False)
                print(f"{embedding} {model_name} {embedding_type} results saved to {os.path.join(results_dir, f'{dataset}_embeddings_train_test_model_results.csv')}")

                #Save model
                model_filename = f"{embedding}_{embedding_type}_{model_name.replace(' ', '_')}.joblib"
                model_path = os.path.join(model_output_dir, model_filename)
                joblib.dump(model, model_path)
                print(f"Model saved to {model_path}")
                    
if __name__ == "__main__":
    main()