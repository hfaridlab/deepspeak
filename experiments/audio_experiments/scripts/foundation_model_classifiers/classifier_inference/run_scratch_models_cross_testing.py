import os
import json
import torch
import numpy as np
import pandas as pd
import yaml
import soundfile as sf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix

def load_config(config_path):
    """Load configuration from a YAML file."""

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def calculate_metrics(y_true, y_score):
    """Calculate evaluation metrics."""

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    y_pred = (y_score >= eer_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    real_acc = tn / (tn + fp)
    fake_acc = tp / (tp + fn)

    return {
        'Acc': acc,
        'AUC': auc,
        'EER': eer,
        'F': f1,
        'real_acc': real_acc,
        'fake_acc': fake_acc
    }

def load_audio(file_path, nb_samp):
    """ Load audio in compatible format for pretrained models, handling padding issues"""

    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if len(audio) < nb_samp:
        audio = np.pad(audio, (0, nb_samp - len(audio)), 'wrap')
    elif len(audio) > nb_samp:
        start = np.random.randint(0, len(audio) - nb_samp)
        audio = audio[start:start + nb_samp]
    return torch.FloatTensor(audio)

def evaluate_model_on_dataset(model, device, config, dataset_path, model_name):
    """Run evaluation on a dataset."""
    
    df = pd.read_csv(dataset_path)
    test_df = df[df['split'] == 'test']

    real_df = test_df[test_df['audio_generator'] == 'real']
    fake_df = test_df[test_df['audio_generator'] != 'real']

    # optional: balance test dataset - have removed as we do not balance test set in the embeddings experiments 
    #min_count = min(len(real_df), len(fake_df))
    #real_df = real_df.sample(min_count, random_state=42)
    #fake_df = fake_df.sample(min_count, random_state=42)

    balanced_df = pd.concat([real_df, fake_df])
    
    scores, labels = [], []
    nb_samp = config['model_config']['nb_samp']

    for _, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        try:
            x = load_audio(row['file_location'], nb_samp).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
                output = out[1] if isinstance(out, tuple) else out
                score = output[0, 0].item() - output[0, 1].item()
                scores.append(score)
                labels.append(1 if row['audio_generator'] == 'real' else 0)
        except Exception:
            continue

    return calculate_metrics(np.array(labels), np.array(scores))

def load_model_and_config(model_info, models_dir):
    """Load model and its configuration."""
    with open(model_info['config']) as f:
        config = json.load(f)

    sys.path.append(models_dir)
    ModelClass = __import__(model_info['class_name'], fromlist=['Model']).Model

    checkpoint = torch.load(config['model_path'], map_location='cpu')
    model = ModelClass(config['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device).eval(), config, device

def main():
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['scratch_cross_testing']
    
    # Load and combine DeepSpeak datasets
    df1 = pd.read_csv(paths['metadata']['deepspeak_v1_1'])[['split', 'file_location', 'audio_generator']]
    df2 = pd.read_csv(paths['metadata']['deepspeak_v2'])[['split', 'file_location', 'audio_generator']]
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Create temporary combined file
    combined_path = os.path.join(os.path.dirname(paths['metadata']['deepspeak_v1_1']), 'temp_deepspeak_combined_test_only.csv')
    combined_df.to_csv(combined_path, index=False)

    datasets = {
        'combined_deepspeak': combined_path,
        'TIMIT_ElevenLabs': paths['metadata']['timit_elevenlabs'],
        'asvspoof': paths['metadata']['asvspoof']
    }

    results = []
    for model_name, model_info in paths['models'].items():
        model, config, device = load_model_and_config(model_info, paths['models_dir'])
        for dataset_name, dataset_path in datasets.items():
            metrics = evaluate_model_on_dataset(model, device, config, dataset_path, model_name)
            results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                **metrics
            })

    # Save results
    output_path = os.path.join(paths['output_dir'], paths['output_file'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    # Clean up temporary file
    if os.path.exists(combined_path):
        os.remove(combined_path)

if __name__ == "__main__":
    import sys
    main()
