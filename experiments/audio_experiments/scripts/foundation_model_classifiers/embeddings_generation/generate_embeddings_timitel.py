import os
import pandas as pd
import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import argparse
import laion_clap
import librosa
from functools import partial
import nemo.collections.asr as nemo_asr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def load_config(config_path):
    """ Load configuration from a YAML file. """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

# Load CLAP model
def load_clap():
    """ Load the LAION CLAP model. """
    
    print("Loading LAION CLAP model...")
    try:
        # Attempt to load the model
        model = laion_clap.CLAP_Module(enable_fusion=False)
        
        # Monkey patch the load function in the library temporarily
        original_torch_load = torch.load
        torch.load = partial(original_torch_load, weights_only=False)
        
        try:
            model.load_ckpt()  # Load with patched torch.load
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
        print("Successfully loaded CLAP model")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Attempting alternative loading method...")
        
        try:
            # Try with a different model variant that might be more compatible
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
            
            # Use the same patching technique
            original_torch_load = torch.load
            torch.load = partial(original_torch_load, weights_only=False)
            
            try:
                model.load_ckpt()
            finally:
                torch.load = original_torch_load
                
            print("Successfully loaded alternative CLAP model")
            return model
        except Exception as e2:
            print(f"All loading attempts failed: {str(e2)}")
            raise

# Extract embeddings from audio file
def extract_embedding_clap(model, file_path):
    """ Extract LAION-CLAP embeddings from audio file."""

    try:
        # Extract embedding using CLAP
        embeddings = model.get_audio_embedding_from_filelist([file_path], use_tensor=False)
        return embeddings.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Process dataset and extract embeddings, saving to CSV
def process_dataset_clap(model, dataset_path, embeddings_dir):
    """ Process dataset and extract embeddings, saving to CSV."""

    print(f"Processing dataset: {os.path.basename(dataset_path)}")
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Generate output CSV filename
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    output_csv = os.path.join(embeddings_dir, f"{dataset_name}_clap_embeddings.csv".replace('_metadata', ''))
    
    # Check if embeddings CSV already exists and has content
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
    
    # Load dataset
    df = pd.read_csv(dataset_path)

    print(f"Loaded {len(df)} samples from {dataset_path}")
    
    # Check for non-string file paths and fix them
    if df['file_location'].dtype != 'object':
        print(f"WARNING: file_location column is not string type ({df['file_location'].dtype}). Converting to string.")
        df['file_location'] = df['file_location'].astype(str)
    
    # Check for NaN values in file_location
    if df['file_location'].isna().any():
        print(f"WARNING: Found {df['file_location'].isna().sum()} NaN values in file_location column. Removing these rows.")
        df = df.dropna(subset=['file_location'])
    
    # Create a new dataframe to store embeddings
    embeddings_data = []
    successful_extractions = 0
    
    # Process each file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        try:
            file_path = str(row['file_location'])  # Ensure file_path is a string
            
            # Skip if file_path is invalid
            if not file_path or file_path == 'nan':
                print(f"Warning: Invalid file path at index {idx}")
                continue
                
            label = 0 if row['audio_generator'] == 'real' else 1  # 0 for real, 1 for fake
            is_training = 1 if row['split'] == 'train' else 0
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            # Extract embedding
            embedding = extract_embedding_clap(model, file_path)
            
            if embedding is not None:
                successful_extractions += 1
                
                # Create a dictionary with metadata and embedding
                embedding_dict = {
                    'file_path': file_path,
                    'label': label,
                    'is_train': is_training,
                    'audio_generator': row['audio_generator']
                }
                
                # Add embedding features
                for i, value in enumerate(embedding):
                    embedding_dict[f'emb_{i}'] = value
                
                embeddings_data.append(embedding_dict)
                
                # Save intermediate results every 100 files to avoid losing progress
                if len(embeddings_data) % 100 == 0:
                    temp_df = pd.DataFrame(embeddings_data)
                    temp_df.to_csv(output_csv + '.temp', index=False)
                    print(f"Saved {len(embeddings_data)} embeddings to temporary file")
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Check if we have any successful extractions
    if len(embeddings_data) == 0:
        print(f"ERROR: No embeddings were successfully extracted from {dataset_path}")
        return None
    
    # Create final dataframe and save to CSV
    embeddings_df = pd.DataFrame(embeddings_data)
    embeddings_df.to_csv(output_csv, index=False)
    
    # Remove temporary file if it exists
    if os.path.exists(output_csv + '.temp'):
        os.remove(output_csv + '.temp')
    
    print(f"Processed {len(embeddings_df)} files. Saved embeddings to {output_csv}")
    return output_csv

# Load TitaNet model
def load_titanet():
    """ Load the TitaNet model. """

    print("Loading TitaNet model...")
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
    return speaker_model

# Extract embeddings from audio file
def extract_embedding_titanet(model, file_path):
    """ Extract TitaNet embeddings from audio file."""

    try:
        # Extract embedding using the exact approach from the working code
        embeddings = model.get_embedding(file_path)
        # Convert to numpy array
        embeddings_np = embeddings.cpu().numpy()
        return embeddings_np.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Process dataset and extract embeddings, saving to CSV
def process_dataset_titanet(model, dataset_path, embeddings_dir):   
    """ Process dataset and extract embeddings, saving to CSV."""

    print(f"Processing dataset: {os.path.basename(dataset_path)}")
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Generate output CSV filename
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    output_csv = os.path.join(embeddings_dir, f"{dataset_name}_titanet_embeddings.csv".replace('_metadata', ''))
    
    # Check if embeddings CSV already exists and has content
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
    
    # Load dataset
    df = pd.read_csv(dataset_path)

    print(f"Loaded {len(df)} samples from {dataset_path}")
    
    # Check for non-string file paths and fix them
    if df['file_location'].dtype != 'object':
        print(f"WARNING: file_location column is not string type ({df['file_location'].dtype}). Converting to string.")
        df['file_location'] = df['file_location'].astype(str)
    
    # Check for NaN values in file_location
    if df['file_location'].isna().any():
        print(f"WARNING: Found {df['file_location'].isna().sum()} NaN values in file_location column. Removing these rows.")
        df = df.dropna(subset=['file_location'])
    
    # Create a new dataframe to store embeddings
    embeddings_data = []
    successful_extractions = 0
    
    # Process each file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        try:
            file_path = str(row['file_location'])  # Ensure file_path is a string
            
            # Skip if file_path is invalid
            if not file_path or file_path == 'nan':
                print(f"Warning: Invalid file path at index {idx}")
                continue
                
            label = 0 if row['audio_generator'] == 'real' else 1  # 0 for real, 1 for fake
            is_training = 1 if row['split'] == 'train' else 0
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            # Extract embedding
            embedding = extract_embedding_titanet(model, file_path)
            
            if embedding is not None:
                successful_extractions += 1
                
                # Create a dictionary with metadata and embedding
                embedding_dict = {
                    'file_path': file_path,
                    'label': label,
                    'is_train': is_training,
                    'audio_generator': row['audio_generator']
                }
                
                # Add embedding features
                for i, value in enumerate(embedding):
                    embedding_dict[f'emb_{i}'] = value
                
                embeddings_data.append(embedding_dict)
                
                # Save intermediate results every 100 files to avoid losing progress
                if len(embeddings_data) % 100 == 0:
                    temp_df = pd.DataFrame(embeddings_data)
                    temp_df.to_csv(output_csv + '.temp', index=False)
                    print(f"Saved {len(embeddings_data)} embeddings to temporary file")
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Check if we have any successful extractions
    if len(embeddings_data) == 0:
        print(f"ERROR: No embeddings were successfully extracted from {dataset_path}")
        return None
    
    # Create final dataframe and save to CSV
    embeddings_df = pd.DataFrame(embeddings_data)
    embeddings_df.to_csv(output_csv, index=False)
    
    # Remove temporary file if it exists
    if os.path.exists(output_csv + '.temp'):
        os.remove(output_csv + '.temp')
    
    print(f"Processed {len(embeddings_df)} files. Saved embeddings to {output_csv}")
    return output_csv


def load_wav2vec2():
    """ Load the XLSR-Wav2Vec2 model. """

    print("Loading XLSR-Wav2Vec2 model...")

    # Error handling: Load just the feature extractor and model without the tokenizer
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, feature_extractor, device

# Extract embeddings from audio file
def extract_embedding_wav2vec(model, feature_extractor, device, file_path):
    """ Extract XLSR-Wav2Vec2 embeddings from audio file."""

    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        
        # Ensure audio is not too short
        if len(audio) < 1000:  # Arbitrary minimum length
            print(f"Warning: Audio file {file_path} is too short. Padding...")
            audio = np.pad(audio, (0, 1000 - len(audio)), 'constant')
        
        # Process audio with Wav2Vec2 feature extractor
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get the hidden states from the last layer
        hidden_states = outputs.last_hidden_state
        
        # Average the hidden states across the time dimension to get a fixed-size embedding
        embedding = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()
        
        return embedding
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Process full dataset and extract embeddings, saving to CSV
def process_dataset_wav2vec(model, feature_extractor, device, dataset_path, embeddings_dir):
    """ Process full dataset and extract embeddings, saving to CSV."""
    
    print(f"Processing dataset: {os.path.basename(dataset_path)}")
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Generate output CSV filename
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    output_csv = os.path.join(embeddings_dir, f"{dataset_name}_xlsrwav2vec2_embeddings.csv".replace('_metadata', ''))
    
    # Check if embeddings CSV already exists and has content
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
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")
    
    # Check for non-string file paths and fix them
    if df['file_location'].dtype != 'object':
        print(f"WARNING: file_location column is not string type ({df['file_location'].dtype}). Converting to string.")
        df['file_location'] = df['file_location'].astype(str)
    
    # Check for NaN values in file_location
    if df['file_location'].isna().any():
        print(f"WARNING: Found {df['file_location'].isna().sum()} NaN values in file_location column. Removing these rows.")
        df = df.dropna(subset=['file_location'])
    
    # Create a new dataframe to store embeddings
    embeddings_data = []
    successful_extractions = 0
    
    # Process each file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        try:
            file_path = str(row['file_location'])  # Ensure file_path is a string
            
            # Skip if file_path is invalid
            if not file_path or file_path == 'nan':
                print(f"Warning: Invalid file path at index {idx}")
                continue
                
            label = 0 if row['audio_generator'] == 'real' else 1  # 0 for real, 1 for fake
            is_training = 1 if row['split'] == 'train' else 0
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            # Extract embedding
            embedding = extract_embedding_wav2vec(model, feature_extractor, device, file_path)
            
            if embedding is not None:
                successful_extractions += 1
                
                # Create a dictionary with metadata and embedding
                embedding_dict = {
                    'file_path': file_path,
                    'label': label,
                    'is_train': is_training,
                    'audio_generator': row['audio_generator']
                }
                
                # Add embedding features
                for i, value in enumerate(embedding):
                    embedding_dict[f'emb_{i}'] = value
                
                embeddings_data.append(embedding_dict)
                
                # Save intermediate results every 100 files to avoid losing progress
                if len(embeddings_data) % 100 == 0:
                    temp_df = pd.DataFrame(embeddings_data)
                    temp_df.to_csv(output_csv + '.temp', index=False)
                    print(f"Saved {len(embeddings_data)} embeddings to temporary file")
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Check if we have any successful extractions
    if len(embeddings_data) == 0:
        print(f"ERROR: No embeddings were successfully extracted from {dataset_path}")
        return None
    
    # Create final dataframe and save to CSV
    embeddings_df = pd.DataFrame(embeddings_data)
    embeddings_df.to_csv(output_csv, index=False)
    
    # Remove temporary file if it exists
    if os.path.exists(output_csv + '.temp'):
        os.remove(output_csv + '.temp')
    
    print(f"Processed {len(embeddings_df)} files. Saved embeddings to {output_csv}")
    return output_csv

def main():
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['embeddings']['timitel']
    
    metadata_path = paths['metadata_path']
    embeddings_dir = paths['output_dir']
    
    # Load CLAP model
    clap_model = load_clap()
    process_dataset_clap(clap_model, metadata_path, embeddings_dir)
    
    # Load titanet model
    titanet_model = load_titanet()
    process_dataset_titanet(titanet_model, metadata_path, embeddings_dir)
    
    # Load wav2vec-xlsr model
    wav2vec_model, feature_extractor, device = load_wav2vec2()
    process_dataset_wav2vec(wav2vec_model, feature_extractor, device, metadata_path, embeddings_dir)

if __name__ == "__main__":
    main()
