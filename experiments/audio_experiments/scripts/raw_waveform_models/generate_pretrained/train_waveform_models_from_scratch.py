import os
import sys
import json
import torch
import pandas as pd
import yaml
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from importlib import import_module
from sklearn.utils import shuffle
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

class DeepSpeakDataset(Dataset):
    """Dataset class for DeepSpeak audio data."""
    
    def __init__(self, df, nb_samp):
        self.df = df.reset_index(drop=True)
        self.nb_samp = nb_samp

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = sf.read(row['file_location'])
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if len(audio) < self.nb_samp:
            audio = np.pad(audio, (0, self.nb_samp - len(audio)), 'wrap')
        elif len(audio) > self.nb_samp:
            start = np.random.randint(0, len(audio) - self.nb_samp)
            audio = audio[start:start + self.nb_samp]
        audio_tensor = torch.FloatTensor(audio)
        label = int(row['audio_source'])
        return audio_tensor, label

def load_balanced_train_data(metadata_paths, nb_samp):
    """Load and balance training data from multiple metadata files."""
    dfs = [pd.read_csv(p) for p in metadata_paths]
    combined = pd.concat(dfs)
    train_df = combined[combined['split'] == 'train'].copy()

    # Labeling - aligned to the pretrained protocol, 0 = real and 1 = fake
    train_df['audio_source'] = train_df['audio_generator'].apply(lambda x: 0 if x == 'real' else 1)

    # Balance real/fake samples
    real = train_df[train_df['audio_source'] == 0]
    fake = train_df[train_df['audio_source'] == 1]
    min_len = min(len(real), len(fake))
    balanced_df = pd.concat([real.sample(min_len, random_state=4), fake.sample(min_len, random_state=42)])
    return shuffle(balanced_df), DeepSpeakDataset(balanced_df, nb_samp)

def train_model(config_path, model_save_dir, metadata_paths):
    """Train model using the specified configuration and data."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['model_config']
    optim_config = config['optim_config']
    nb_samp = model_config['nb_samp']
    batch_size = config.get("batch_size", 16)
    epochs = 50  # Fixed at 50 epochs for training efficiency

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_name = model_config['architecture']
    model_module = import_module(f"models.{arch_name}")
    model = model_module.Model(model_config).to(device)

    # Dataset setup
    df_balanced, train_dataset = load_balanced_train_data(metadata_paths, nb_samp)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss setup
    lr = optim_config.get("lr", optim_config.get("base_lr", 0.0001))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)[1] if isinstance(model(batch_x), tuple) else model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

    # Save model
    model_type = os.path.splitext(os.path.basename(config_path))[0]
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{model_type}_deepspeak_from_scratch.pth")
    torch.save(model.state_dict(), model_path)

def main():
    repo_config = load_config('repo_config.yaml')
    if repo_config is None:
        raise ValueError("Failed to load config file")
    
    paths = repo_config['paths']['model_training']
    
    # Train each model type
    for model_name, config_path in paths['configs'].items():
        train_model(
            config_path,
            paths['output_dir'],
            [paths['metadata']['v1_1'], paths['metadata']['v2']]
        )

if __name__ == "__main__":
    sys.path.append("/home/ubuntu/DeepSpeak/aasist")
    main()

# Evaluation metrics on test set, aligned to pretrained and embeddings experiments 
def evaluate_model(model, test_loader, device):
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).unsqueeze(1)
            y = y.numpy()
            outputs = model(x)
            outputs = outputs[1] if isinstance(outputs, tuple) else outputs
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y)

    y_true = np.array(all_labels)
    y_prob = np.array(all_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    y_pred = (y_prob >= eer_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
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

   