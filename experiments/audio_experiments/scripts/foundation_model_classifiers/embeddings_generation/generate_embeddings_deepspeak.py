import os
import pandas as pd
import numpy as np
import torch
import yaml
from tqdm import tqdm
import nemo.collections.asr as nemo_asr

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

def load_titanet():
    """Load TitaNet model."""
    print("Loading TitaNet model...")
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
    return speaker_model

def extract_embedding(model, file_path):
    """Extract embeddings from audio file."""
    try:
        embeddings = model.get_embedding(file_path)
        embeddings_np = embeddings.cpu().numpy()
        return embeddings_np.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_dataset(model, dataset_path, embeddings_dir, version):
    """Process dataset and extract embeddings, saving to CSV."""
    print(f"Processing DeepSpeak {version} dataset: {os.path.basename(dataset_path)}")
    
    os.makedirs(embeddings_dir, exist_ok=True)
    
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    output_csv = os.path.join(embeddings_dir, f"{dataset_name}_titanet_embeddings.csv".replace('_metadata', ''))
    
    if os.path.exists(output_csv):
        try:
            df_check = pd.read_csv(output_csv)
            if len(df_check) > 0:
                print(f"Loading existing embeddings from {output_csv} ({len(df_check)} samples)")
                return output_csv
            else:
                print(f"Existing file {output_csv} is empty. Recreating...")
        except Exception as e:
            print(f"Error reading existing file {output_csv}: {str(e)}. Recreating...")
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")
    
    if df['file_location'].dtype != 'object':
        print(f"WARNING: file_location column is not string type ({df['file_location'].dtype}). Converting to string.")
        df['file_location'] = df['file_location'].astype(str)
    
    if df['file_location'].isna().any():
        print(f"WARNING: Found {df['file_location'].isna().sum()} NaN values in file_location column. Removing these rows.")
        df = df.dropna(subset=['file_location'])
    
    embeddings_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting embeddings for {version}"):
        try:
            file_path = str(row['file_location'])
            
            if not file_path or file_path == 'nan':
                print(f"Warning: Invalid file path at index {idx}")
                continue
                
            label = 0 if row['audio_generator'] == 'real' else 1
            is_training = 1 if row['split'] == 'train' else 0
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            embedding = extract_embedding(model, file_path)
            
            if embedding is not None:
                embedding_dict = {
                    'file_path': file_path,
                    'label': label,
                    'is_train': is_training,
                    'audio_generator': row['audio_generator']
                }
                
                for i, value in enumerate(embedding):
                    embedding_dict[f'emb_{i}'] = value
                
                embeddings_data.append(embedding_dict)
                
                if len(embeddings_data) % 100 == 0:
                    temp_df = pd.DataFrame(embeddings_data)
                    temp_df.to_csv(output_csv + '.temp', index=False)
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    if len(embeddings_data) == 0:
        print(f"ERROR: No embeddings were successfully extracted from {dataset_path}")
        return None
    
    embeddings_df = pd.DataFrame(embeddings_data)
    embeddings_df.to_csv(output_csv, index=False)
    
    if os.path.exists(output_csv + '.temp'):
        os.remove(output_csv + '.temp')
    
    print(f"Processed {len(embeddings_df)} files for {version}. Saved embeddings to {output_csv}")
    return output_csv

def main():
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['embeddings']['deepspeak']
    
    # Load TitaNet model
    model = load_titanet()
    
    # Process v1.1 and v2 datasets separately
    v1_1_embeddings_csv = process_dataset(model, paths['v1_1_metadata_path'], paths['output_dir'], 'v1.1')
    v2_embeddings_csv = process_dataset(model, paths['v2_metadata_path'], paths['output_dir'], 'v2')
    
    if v1_1_embeddings_csv is None or v2_embeddings_csv is None:
        print("ERROR: Failed to generate embeddings for one or both versions")
        return

    print("\nEmbedding generation completed successfully")
    print(f"v1.1 embeddings saved to: {v1_1_embeddings_csv}")
    print(f"v2 embeddings saved to: {v2_embeddings_csv}")

if __name__ == "__main__":
    main() 