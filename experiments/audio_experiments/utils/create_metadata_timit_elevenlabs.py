import os
import csv
import random
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

def create_metadata_csv(base_dir, output_csv):
    """Create a metadata CSV from the TIMIT_ElevenLabs dataset."""
    
    print('starting')

    # Delete the output CSV if it exists
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Existing metadata CSV deleted: {output_csv}")
    
    # Define CSV columns
    columns = [
        'split', 'type', 'video_file', 'file_location', 'kind', 'engine',
        'identity_source', 'identity_target', 'recording_source', 'recording_target',
        'recording_target_ai_generated', 'gesture_type', 'script_type', 'real_recording',
        'real_question_number', 'real_transcript', 'real_identity', 'audio_generator'
    ]
    
    metadata = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.flac', '.wav', '.mp3')):
                file_path = os.path.join(root, file)
                audio_generator = 'real' if 'real' in root else 'fake'
                split = 'train' if random.random() < 0.8 else 'test'
                
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
                if len(metadata) % 10 == 0:
                    with open(output_csv, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=columns)
                        if csvfile.tell() == 0:  # Check if the file is empty to write the header
                            writer.writeheader()
                        writer.writerows(metadata)
                    print(f'Metadata saved to {output_csv}')
                    metadata = []  # Clear metadata after writing

if __name__ == "__main__":
    
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['timit_elevenlabs']
    
    base_dir = paths['base_dir']
    output_csv = paths['output_csv']
    
    create_metadata_csv(base_dir, output_csv)