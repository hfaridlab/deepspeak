import argparse
import json
import os
import subprocess
import yaml
from tqdm import tqdm

def load_config(config_path):
    """ Load configuration from a JSON or YAML file. """
    
    try:
        file_ext = os.path.splitext(config_path)[1].lower()
        with open(config_path, 'r') as f:
            if file_ext == '.json':
                config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
        return config
    except Exception:
        return None

def extract_raw_audio_from_mp4(input_file, output_file):
    """ Extract raw audio from MP4 file without resampling. """
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian format
            '-y',  # DO overwrite output file if it exists
            output_file
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, Exception):
        return False

def resample_audio(input_file, output_file, sample_rate=16000):
    """ Resample audio file to specified sample rate. """

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, Exception):
        return False

def process_folder(input_folder, output_folder, real_identifier="real", fake_identifier="fake", sample_rate=16000):
    """
    Process all MP4 files in the input folder in two steps:
    1. Extract raw audio
    2. Resample to specified sample rate
    
    Preserves train/test structure in the output folders. 
    """

    raw_dir = os.path.join(output_folder, "raw")
    processed_dir = os.path.join(output_folder, f"processed_{sample_rate}Hz")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    mp4_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    
    extraction_success = 0
    processing_success = 0
    
    for mp4_file in tqdm(mp4_files, desc=f"Processing {input_folder}"):
        is_real = real_identifier in mp4_file.lower() and fake_identifier not in mp4_file.lower()
        rel_path = os.path.relpath(os.path.dirname(mp4_file), input_folder)
        filename = os.path.basename(mp4_file).replace('.mp4', '.wav')
        
        raw_output_dir = os.path.join(raw_dir, rel_path)
        processed_output_dir = os.path.join(processed_dir, rel_path)
        
        os.makedirs(raw_output_dir, exist_ok=True)
        os.makedirs(processed_output_dir, exist_ok=True)
        
        raw_output_file = os.path.join(raw_output_dir, filename)
        if os.path.exists(raw_output_file):
            continue
        processed_output_file = os.path.join(processed_output_dir, filename)
        
        if extract_raw_audio_from_mp4(mp4_file, raw_output_file):
            extraction_success += 1
            if resample_audio(raw_output_file, processed_output_file, sample_rate):
                processing_success += 1
    
    return processing_success, len(mp4_files)

def main():
    parser = argparse.ArgumentParser(description='Extract and process audio from MP4 files in two steps.')
    parser.add_argument('--input_folders', nargs='+', required=True, 
                        help='List of input folders containing MP4 files')
    parser.add_argument('--output_folders', nargs='+', required=True,
                        help='List of output folders where WAV files will be saved')
    parser.add_argument('--real_identifier', default='real', 
                        help='String identifier for real audio files')
    parser.add_argument('--fake_identifier', default='fake', 
                        help='String identifier for fake audio files')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target sample rate in Hz for processed audio')
    
    args = parser.parse_args()
    
    if len(args.input_folders) != len(args.output_folders):
        return
    
    total_success = 0
    total_files = 0
    
    for input_folder, output_folder in zip(args.input_folders, args.output_folders):
        success, total = process_folder(
            input_folder, 
            output_folder,
            args.real_identifier,
            args.fake_identifier,
            args.sample_rate
        )
        total_success += success
        total_files += total

if __name__ == "__main__":
    main()