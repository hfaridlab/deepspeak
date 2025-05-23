import os
import glob
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

def load_config(config_path):
    """ Load configuration from a YAML file. """

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def main():
    # Load configuration
    config = load_config('repo_config.yaml')
    if config is None:
        raise ValueError("Failed to load config file")
    
    paths = config['paths']['cosine_pairs']
    
    # 1. Load all titanet embedding CSVs
    embedding_files = glob.glob(os.path.join(paths['embeddings_dir'], '*titanet*.csv'))
    
    embedding_dfs = []
    for file in embedding_files:
        df = pd.read_csv(file)
        df['embedding_source_file'] = os.path.basename(file)
        embedding_dfs.append(df)
    embeddings_df = pd.concat(embedding_dfs, ignore_index=True)
    
    # 2. Load all deepspeak metadata CSVs
    metadata_files = glob.glob(os.path.join(paths['metadata_dir'], '*deepspeak*.csv'))
    
    metadata_dfs = []
    for file in metadata_files:
        df = pd.read_csv(file)
        df['metadata_source_file'] = os.path.basename(file)
        metadata_dfs.append(df)
    metadata_df = pd.concat(metadata_dfs, ignore_index=True)
    
    # Filter only real 
    metadata_df[metadata_df['type'] == 'real']
    
    # 3. For each identity, select a random "scripted-short" row
    def get_identity(row):
        return row['identity_source'] if pd.notnull(row.get('identity_source', None)) and row['identity_source'] != '' else row.get('real_identity', None)
    
    metadata_df['identity'] = metadata_df.apply(get_identity, axis=1)
    
    def is_real_question_number_four(row):
        real_question_number = row.get('real_question_number', '')
        if isinstance(real_question_number, float) or pd.isna(real_question_number):
            real_question_number = str(real_question_number)

        # Take same input sentence for all. Include a fallback by way of error handling, using a similar sentence in length. 
        # In this implementation, we only see this in 3 cases.    
        if real_question_number == '4.0_stan_sent_qn.8':
            return True
        elif '4.0_stan_sent_qn' in real_question_number:
            return True
        return False
    
    metadata_df = metadata_df[metadata_df.apply(is_real_question_number_four, axis=1)]
    metadata_sampled = metadata_df.groupby('identity').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
    
    # 4. Match audio file to embeddings
    def get_audio_basename(row):
        video_file = row['video_file']
        return os.path.splitext(os.path.basename(video_file))[0]
    
    metadata_sampled['audio_basename'] = metadata_sampled.apply(get_audio_basename, axis=1)
    
    # Find matching embedding row for each identity
    def find_embedding_row(audio_basename):
        matches = embeddings_df[embeddings_df['file_path'].str.contains(audio_basename, na=False)]
        if not matches.empty:
            return matches.iloc[0]
        else:
            return pd.Series([np.nan]*len(embeddings_df.columns), index=embeddings_df.columns)
    
    embedding_rows = metadata_sampled['audio_basename'].apply(find_embedding_row)
    embedding_rows = embedding_rows.reset_index(drop=True)
    
    # 5. Merge metadata and embeddings
    final_df = pd.concat([metadata_sampled.reset_index(drop=True), embedding_rows.reset_index(drop=True)], axis=1)
    
    # 6. Compute cosine similarity and find closest pairs
    emb_cols = [col for col in final_df.columns if col.startswith('emb_')]
    embeddings = final_df[emb_cols].values
    
    def calculate_distances_and_pairs(embeddings):
        distances = cosine_similarity(embeddings)
        np.fill_diagonal(distances, -np.inf)
        all_pairs = [(i, j, distances[i, j]) for i in range(len(embeddings)) for j in range(i + 1, len(embeddings))]
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)
        paired_indices = set()
        final_pairs = []
        actual_sims = []
        for i, j, sim in sorted_pairs:
            if i not in paired_indices and j not in paired_indices:
                final_pairs.append((i, j))
                paired_indices.add(i)
                paired_indices.add(j)
                actual_sims.append(sim)
        return final_pairs, actual_sims
    
    pairs, sims = calculate_distances_and_pairs(embeddings)
    
    # Save the merged dataframe and pairs
    os.makedirs(paths['output_dir'], exist_ok=True)
    final_df.to_csv(os.path.join(paths['output_dir'], 'merged_metadata_embeddings.csv'), index=False)
    
    pairs_df = pd.DataFrame([{
        'identity_1': str(int(final_df.iloc[i]['identity'])),
        'identity_2': str(int(final_df.iloc[j]['identity'])),
        'similarity': sim
    } for (i, j), sim in zip(pairs, sims)])
    pairs_df.to_csv(os.path.join(paths['output_dir'], 'closest_identity_pairs.csv'), index=False)
    
    audio_pairs_df = pairs_df.drop(columns=['similarity']).reset_index(drop=True)
    audio_pairs_df.to_csv(os.path.join(paths['output_dir'], 'audio_identity_pairs.csv'), index=False)
    
    print("Done! Merged data and closest pairs saved.")
    print(f"Total unique participants (identities): {final_df['identity'].nunique()}")
    print(f"Total rows in merged DataFrame: {len(final_df)}")
    print(f"Total pairs found: {len(pairs)}")

if __name__ == "__main__":
    main()