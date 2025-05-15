import os
import csv
import glob
import yaml

def load_config(config_path):
    """ Load configuration from a YAML file. """

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def build_file_cache(base_dir):
    """Build a cache of all audio files in the base directory."""

    file_cache = {}
    print(f"Building file cache for {base_dir}...")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.flac'):
                file_cache[file] = os.path.join(root, file)
    print(f"File cache built with {len(file_cache)} files")
    return file_cache

def find_audio_file(file_cache, audio_file_name):
    """Find an audio file using the cache. """

    return file_cache.get(audio_file_name)

def parse_protocol_file(protocol_file, base_dir, output_csv,columns, file_cache):
    """Parse a protocol file and extract metadata."""
    
    metadata = []
    split = 'train' if 'trn' in protocol_file else 'test' if 'dev' in protocol_file else 'eval'
    
    with open(protocol_file, 'r') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)  
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            audio_file_name = parts[1] + '.flac'
            audio_generator = 'real' if parts[-1] == 'bonafide' else 'fake'
            if i % 10 == 0:
                print(f'Searching for {audio_file_name} in {base_dir} (file {i} out of {total_lines})')
            file_path = find_audio_file(file_cache, audio_file_name)
            if file_path:
                metadata.append({
                    'split': split,
                    'type': '',
                    'video_file': '',
                    'file_location': file_path,
                    'kind': '',
                    'engine': '',
                    'identity_source': '',
                    'identity_target': '',
                    'recording_source': '',
                    'recording_target': '',
                    'recording_target_ai_generated': '',
                    'gesture_type': '',
                    'script_type': '',
                    'real_recording': '',
                    'real_question_number': '',
                    'real_transcript': '',
                    'real_identity': '',
                    'audio_generator': audio_generator
                })

             # Write to CSV every 10 files
            if i % 10 == 0 or i == total_lines:
                with open(output_csv, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=columns)
                    if csvfile.tell() == 0:  # Check if the file is empty to write the header
                        writer.writeheader()
                    writer.writerows(metadata)
                print(f'Metadata saved to {output_csv}')
                metadata = []  # Clear metadata after writing
  
    return metadata

def create_metadata_csv(protocol_dir, base_dir, output_csv):
    """Create a metadata CSV from protocol files."""

    # Delete the output CSV if it exists
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Existing metadata CSV deleted: {output_csv}")

    all_metadata = []
    protocol_files = glob.glob(os.path.join(protocol_dir, '*.txt'))

    # Define CSV columns
    columns = [
        'split', 'type', 'video_file', 'file_location', 'kind', 'engine',
        'identity_source', 'identity_target', 'recording_source', 'recording_target',
        'recording_target_ai_generated', 'gesture_type', 'script_type', 'real_recording',
        'real_question_number', 'real_transcript', 'real_identity', 'audio_generator'
    ]
    # Build file cache once
    file_cache = build_file_cache(base_dir) 

    for protocol_file in protocol_files:
        parse_protocol_file(protocol_file, base_dir, output_csv, columns, file_cache)

if __name__ == "__main__":
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['asvspoof']
    
    protocol_dir = paths['protocol_dir']
    base_dir = paths['base_dir']
    output_csv = paths['output_csv']
    
    create_metadata_csv(protocol_dir, base_dir, output_csv)