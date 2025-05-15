import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def build_file_index(base_dir):
    """ Build an index of all audio files in the base directory. """

    print(f"Building file index from {base_dir}...")
    start_time = time.time()
    
    file_index = {}
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    # Walk through the directory tree
    for root, _, files in os.walk(base_dir):
        for file in files:
            # Check if it's an audio file
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                # Get filename without extension
                filename_without_ext = os.path.splitext(file)[0]
                
                # Store the full path
                file_index[filename_without_ext] = os.path.join(root, file)
                
                # Also store alternate versions/lowercase of filenames for more flexible matching
                file_index[filename_without_ext.lower()] = os.path.join(root, file)
                
                # 2. Store without any specific generator tag at the end
                if '-' in filename_without_ext:
                    base_name = filename_without_ext.rsplit('-', 1)[0]
                    if base_name not in file_index:  # Don't overwrite if exists
                        file_index[base_name] = os.path.join(root, file)
    
    elapsed_time = time.time() - start_time
    print(f"Found {len(file_index)} audio files in {elapsed_time:.2f} seconds")
    return file_index

def find_alternative_file(filename, file_index):
    """ Try different variations of the filename to find a match. """

    # 1. Try the original filename
    if filename in file_index:
        return file_index[filename]
    
    # 2. Try lowercase
    if filename.lower() in file_index:
        return file_index[filename.lower()]
    
    # 3. Try with 'real' suffix
    if '-' in filename:
        parts = filename.rsplit('-', 1)
        real_filename = f"{parts[0]}-real"
        if real_filename in file_index:
            return file_index[real_filename]
    
    # 4. Try without any suffix
    if '-' in filename:
        base_name = filename.rsplit('-', 1)[0]
        if base_name in file_index:
            return file_index[base_name]
    
    # 5. Try just the numeric part if it exists
    import re
    numbers = re.findall(r'\d+', filename)
    if numbers:
        for num in numbers:
            if num in file_index:
                return file_index[num]
    
    # 6. Look for any file that contains this filename
    for key in file_index:
        if filename in key:
            return file_index[key]
    
    return None

def create_metadata_csv(dataset_name, output_path, file_index):
    """ Create a metadata CSV file from a DeepSpeak dataset. """

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    all_metadata = []
    missing_files = []
    
    # Process each split (train and test)
    for split in dataset.keys():
        print(f"Processing {split} split...")
        
        # Loop through all examples in the split
        for example in tqdm(dataset[split]):
            metadata_entry = {
                'split': split,
                'type': example.get('type', None)
            }
            
            # Extract video file path if available
            if 'video-file' in example:
                video_filename = os.path.basename(example['video-file'])
                
                metadata_entry['video_file'] = video_filename
                
                # Get filename without extension for audio file search
                filename_without_ext = os.path.splitext(video_filename)[0]
                
                # Try multiple strategies to find the file
                file_location = find_alternative_file(filename_without_ext, file_index)
                
                if file_location is None:
                    # Log the missing file
                    missing_files.append(filename_without_ext)
                
                metadata_entry['file_location'] = file_location
            
            # Extract metadata from metadata-fake if available
            if 'metadata-fake' in example and example['metadata-fake'] is not None:
                for key, value in example['metadata-fake'].items():
                    # Replace hyphens with underscores for column names
                    metadata_entry[key.replace('-', '_')] = value
            
            # Extract metadata from metadata-real if available
            if 'metadata-real' in example and example['metadata-real'] is not None:
                for key, value in example['metadata-real'].items():
                    # Add prefix to avoid column name conflicts
                    metadata_entry[f"real_{key}"] = value
            
            # Add audio_generator column based on dataset version
            if 'deepspeak_v1_1' in dataset_name:
                # For v1_1: Use recording_target_ai_generated
                if metadata_entry.get('recording_target_ai_generated') == 'True':
                    metadata_entry['audio_generator'] = 'elevenlabs'
                else:
                    metadata_entry['audio_generator'] = 'real'
            elif 'deepspeak_v2' in dataset_name:
                # For v2: Use audio_config
                metadata_entry['audio_generator'] = metadata_entry.get('audio_config')
                # If audio_config is None, set audio_generator to 'real'
                if metadata_entry.get('audio_generator') is None:
                    metadata_entry['audio_generator'] = 'real'
            
            all_metadata.append(metadata_entry)
    
    df = pd.DataFrame(all_metadata)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    # Save list of missing files
    missing_files_path = output_path.replace('.csv', '_missing_files.txt')
    with open(missing_files_path, 'w') as f:
        f.write('\n'.join(missing_files))
    
    print(f"Metadata saved to: {output_path}")
    print(f"Total entries: {len(df)}")
    print(f"Files found: {df['file_location'].notna().sum()}")
    print(f"Files not found: {df['file_location'].isna().sum()}")
    print(f"List of missing files saved to: {missing_files_path}")
    
    # Print column names
    print(f"Columns: {', '.join(df.columns)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create metadata CSVs from DeepSpeak datasets")
    parser.add_argument("--output_dir", type=str, default="./metadata",
                        help="Directory to save metadata CSV files")
    parser.add_argument("--audio_dir", type=str, default="../audio_only",
                        help="Base directory to search for audio files")
    parser.add_argument("--cache_index", action="store_true",
                        help="Cache the file index to disk for faster subsequent runs")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use file index cache to save recursive search time
    index_cache_path = os.path.join(args.output_dir, "file_index_cache.pkl")
    if args.cache_index and os.path.exists(index_cache_path):
        print(f"Loading file index from cache: {index_cache_path}")
        file_index = pd.read_pickle(index_cache_path)
    else:
        # Build the file index once
        file_index = build_file_index(args.audio_dir)
        
        # Cache the index if requested
        if args.cache_index:
            print(f"Caching file index to: {index_cache_path}")
            pd.to_pickle(file_index, index_cache_path)
    
    # Process DeepSpeak v1.1
    v1_1_output = os.path.join(args.output_dir, "deepspeak_v1_1_metadata.csv")
    create_metadata_csv("faridlab/deepspeak_v1_1", v1_1_output, file_index)
    
    # Process DeepSpeak v2
    v2_output = os.path.join(args.output_dir, "deepspeak_v2_metadata.csv")
    create_metadata_csv("faridlab/deepspeak_v2", v2_output, file_index)
    
    print("\nMetadata extraction complete!")

if __name__ == "__main__":
    main()