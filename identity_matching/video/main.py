
import os
import random
import ast
import torch
import clip
import argparse

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from pairing.utils import extract_nth_frame, load_config
from pairing.visualization import create_and_save_pair_images
from orchestration.utils import load_config, get_nested


def preprocess_data(df_users, df_recordings, dataset_base_path, num_videos, frame_choices,
                    excluded_participant_ids=None):
    """
    Preprocesses data by extracting video paths for valid participants and recordings.

    :param df_users: dataframe containing user information
    :param df_recordings: dataframe containing video recording annotations
    :param dataset_base_path: base directory of the dataset with mp4 videos
    :param num_videos: number of videos to process per user
    :param frame_choices: comma-separated frame indices to sample from videos
    :param excluded_participant_ids: list of participant identifiers to exclude from the dataset
    :return: list of video paths per user, list of user IDs, and list of frame choices
    """

    id_img_paths = []
    user_ids = []
    frame_choices = list(map(int, frame_choices.split(",")))

    print("[1/3] preprocessing starting...")

    for _, row in df_users.iterrows():
        curr_img_path = []
        num_valid = 0

        if row["status"] != "APPROVED":
            print("\t", "> dropping", row["server_recording_id"], "because it is not approved")
            continue

        if excluded_participant_ids:
            if row["server_recording_id"] in excluded_participant_ids:
                print("\t", "> dropping", row["server_recording_id"], "because it is explicitly excluded")
                continue

        for record in random.sample(ast.literal_eval(row["valid_recordings"]), len(ast.literal_eval(row["valid_recordings"]))):
            if num_valid >= num_videos:
                break

            filtered_sample = df_recordings[
                df_recordings["recording"] == row["server_recording_id"] + "_" + record + ".mp4"
            ]["question_number"].to_list()
            if not filtered_sample:
                continue

            curr_img_path.append(
                os.path.join(dataset_base_path, filtered_sample[0], row["server_recording_id"] + "_" + record + ".mp4")
            )
            num_valid += 1

        if curr_img_path:
            id_img_paths.append(curr_img_path)
            user_ids.append(row["server_recording_id"])
        else:
            print("\t", "> dropping", row["server_recording_id"], "due to no valid recordings found for embedding")

    print("[2/3] preprocessing completed")
    print("\t", "fin len:", len(user_ids), "out of", df_users.shape[0], "(", len(user_ids) / df_users.shape[0], ")")
    return id_img_paths, user_ids, frame_choices


def compute_embeddings(id_img_paths, frame_choices, preprocess, model, device):
    """
    Computes image embeddings for participants' videos using the specified CLIP model.

    :param id_img_paths: list of video paths for each user
    :param frame_choices: list of frame indices to sample from videos
    :param preprocess: preprocessing function for the CLIP model
    :param model: CLIP model used to generate embeddings
    :param device: device for model inference (cpu, cuda, mps)
    :return: original video frames as PIL images and computed image embeddings
    """

    orig_pils = [[extract_nth_frame(path, random.choice(frame_choices)) for path in paths] for paths in id_img_paths]
    id_img_pils = [[preprocess(path.copy()).unsqueeze(0).to(device) for path in paths if path][:5] for paths in orig_pils]

    with torch.no_grad():
        id_img_embs = [[model.encode_image(img_pil).cpu().squeeze(0) for img_pil in img_pil_batch] for img_pil_batch in id_img_pils]
        id_img_embs = [torch.stack(img_pil_batch).mean(dim=0) for img_pil_batch in id_img_embs]

    return orig_pils, id_img_embs


def calculate_distances_and_pairs(embeddings):
    """
    Calculates cosine similarity between embeddings and determines the best pairs.

    :param embeddings: list of image embeddings
    :return: list of paired indices and actual similarity scores
    """

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


def save_results(final_pairs, user_ids, output_csv):
    """
    Saves the paired participants' IDs to a CSV file.

    :param final_pairs: list of paired participant indices
    :param user_ids: list of user IDs
    :param output_csv: path to save the output CSV file
    """

    df_output = pd.DataFrame(columns=["identity_0", "identity_1"])

    for a, b in final_pairs:
        df_output.loc[len(df_output.index)] = [user_ids[a], user_ids[b]]
        df_output.loc[len(df_output.index)] = [user_ids[b], user_ids[a]]

    df_output.to_csv(output_csv, index=False)


def run(participant_lookup_csv_path, recording_csv_path, dataset_base_dir, participant_pairs_csv_path,
        participant_pairs_visualization_dir, model_name, device, num_videos, frame_choices, excluded_participant_ids):
    """
    Runs the pipeline for pairing participants based on their appearance.

    :param participant_lookup_csv_path: path to a lookup CSV with all the participants (likely master_user_lookup.csv)
    :param recording_csv_path: path to a CSV with all the recording annotations (likely transcript_question_number_mapper.csv)
    :param dataset_base_dir: base directory of the dataset with the source (not deepfaked) mp4 videos
    :param participant_pairs_csv_path: path to save the output CSV file with participant pairs
    :param participant_pairs_visualization_dir: directory to save the pair visualizations
    :param model_name: CLIP model used for embedding generation
    :param device: device for model inference (cpu, cuda, mps)
    :param num_videos: number of videos to process per user (whose embeddings will be averaged)
    :param frame_choices: comma-separated frame indices to sample from videos
    :param excluded_participant_ids: list of participant identifiers to exclude from the dataset
    """

    df_users = pd.read_csv(participant_lookup_csv_path)
    df_recordings = pd.read_csv(recording_csv_path)

    model, preprocess = clip.load(model_name, device=device)
    dataset_dir = os.path.join(dataset_base_dir, "video_mp4")

    id_img_paths, user_ids, frame_choices = preprocess_data(df_users, df_recordings, dataset_dir, num_videos,
                                                            frame_choices, excluded_participant_ids)
    orig_pils, id_img_embs = compute_embeddings(id_img_paths, frame_choices, preprocess, model, device)
    embeddings = torch.stack(id_img_embs, dim=0)

    final_pairs, _ = calculate_distances_and_pairs(embeddings)
    save_results(final_pairs, user_ids, participant_pairs_csv_path)
    create_and_save_pair_images(orig_pils, final_pairs, participant_pairs_visualization_dir)

    print("[3/3] saving and viz completed")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/Users/matyasbohacek/Projects/deepspeak-orchestrator/config/v2.yaml", help="path to the configuration yaml file")
    parser.add_argument("--participant_lookup_csv_path", type=str, help="path to a lookup csv with all the participants")
    parser.add_argument("--recording_csv_path", type=str, help="path to a csv with all the recording annotations")
    parser.add_argument("--dataset_base_dir", type=str, help="base directory of the dataset with the source mp4 videos")
    parser.add_argument("--participant_pairs_csv_path", type=str, help="path to save the output csv file with participant pairs")
    parser.add_argument("--participant_pairs_visualization_dir", type=str, help="directory to save the pair visualizations")
    parser.add_argument("--model_name", type=str, help="clip model used for embedding generation")
    parser.add_argument("--device", type=str, help="device for model inference (cpu, cuda, mps)")
    parser.add_argument("--num_videos", type=int, help="number of videos to process per user")
    parser.add_argument("--frame_choices", type=str, help="comma-separated frame indices to sample from videos")
    parser.add_argument("--excluded_participant_ids", type=str, help="comma-separated list of participant identifiers to exclude")
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    final_args = {
        "participant_lookup_csv_path": args.participant_lookup_csv_path or get_nested(config, "pairing.paths.participant_lookup_csv_path"),
        "recording_csv_path": args.recording_csv_path or get_nested(config, "pairing.paths.recording_csv_path"),
        "dataset_base_dir": args.dataset_base_dir or get_nested(config, "pairing.paths.dataset_base_dir"),
        "participant_pairs_csv_path": args.participant_pairs_csv_path or get_nested(config, "pairing.paths.participant_pairs_csv_path"),
        "participant_pairs_visualization_dir": args.participant_pairs_visualization_dir or get_nested(config, "pairing.paths.participant_pairs_visualization_dir"),
        "model_name": args.model_name or get_nested(config, "pairing.params.model_name"),
        "device": args.device or get_nested(config, "pairing.params.device"),
        "num_videos": args.num_videos or get_nested(config, "pairing.params.num_videos"),
        "frame_choices": args.frame_choices or get_nested(config, "pairing.params.frame_choices"),
        "excluded_participant_ids": args.excluded_participant_ids.split(",") if args.excluded_participant_ids else get_nested(config, "pairing.params.excluded_participant_ids")
    }

    return final_args


if __name__ == "__main__":
    args = parse_arguments()
    run(
        participant_lookup_csv_path=args["participant_lookup_csv_path"],
        recording_csv_path=args["recording_csv_path"],
        dataset_base_dir=args["dataset_base_dir"],
        participant_pairs_csv_path=args["participant_pairs_csv_path"],
        participant_pairs_visualization_dir=args["participant_pairs_visualization_dir"],
        model_name=args["model_name"],
        device=args["device"],
        num_videos=args["num_videos"],
        frame_choices=args["frame_choices"],
        excluded_participant_ids=args['excluded_participant_ids']
    )
