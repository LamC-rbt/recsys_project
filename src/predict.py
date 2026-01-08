import logging
from argparse import ArgumentParser
from pathlib import Path
import csv

import torch
from torch import nn

from recsys_project.configs.config import SASRecConfig, SASRecTrainingConfig
from recsys_project.utils.dataset_utils import get_num_items
from recsys_project.utils.general_utils import build_model, get_device, load_config


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    map_location: torch.device,
) -> None:
    """
    Loads model weights from a local checkpoint file if present.
    """
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint does not exist: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    logger.info(f"Loading model state dict from local checkpoint: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location=map_location)
    model.load_state_dict(state)
    logger.info("Successfully loaded local checkpoint.")


def read_sequences_from_csv(input_path: Path) -> list[list[int]]:
    """
    Read sequences from a CSV-like file.

    Each line stores a (possibly different) number of integers separated by ','.
    Example line:  1,5,22,7
    """
    sequences: list[list[int]] = []

    with input_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for line_idx, row in enumerate(reader, start=1):
            # row is already split on ","
            tokens = [x.strip() for x in row if x.strip() != ""]
            if not tokens:
                logger.warning(f"Line {line_idx} is empty or has no valid integers; skipping.")
                continue

            try:
                seq = [int(x) for x in tokens]
            except ValueError as e:
                logger.error(f"Failed to parse integers on line {line_idx}: {row}. Error: {e}")
                raise

            sequences.append(seq)

    logger.info(f"Loaded {len(sequences)} sequences from {input_path}")
    return sequences


def generate_topk_predictions(
    model: nn.Module,
    sequences: list[list[int]],
    max_seq_len: int,
    top_k: int,
    device: torch.device,
) -> list[list[int]]:
    """
    For each sequence:
      - keep only the last `max_seq_len` interactions
      - turn into tensor
      - add batch dimension
      - get top-k recommendations
    Returns list of predicted item-id lists (one list per input sequence).
    """
    model.eval()
    predictions: list[list[int]] = []

    with torch.no_grad():
        for idx, seq in enumerate(sequences):
            if not seq:
                logger.warning(f"Sequence {idx} is empty; generating empty predictions.")
                predictions.append([])
                continue

            # Slice last max_seq_len interactions
            if len(seq) > max_seq_len:
                truncated_seq = seq[-max_seq_len:]
            else:
                truncated_seq = seq

            # Convert to tensor and add batch dimension: shape (1, L)
            input_tensor = torch.tensor(truncated_seq, dtype=torch.long, device=device).unsqueeze(0)

            # Exclude already-rated items from recommendations
            rated = [truncated_seq]

            # Model returns (indices, scores); we only need indices
            top_indices, _ = model.get_predictions(input_tensor, limit=top_k, rated=rated)

            # Remove batch dimension and move to CPU list
            preds = top_indices.squeeze(0).cpu().tolist()
            predictions.append(preds)

    logger.info(f"Generated predictions for {len(sequences)} sequences.")
    return predictions


def write_predictions_to_csv(predictions: list[list[int]], output_path: Path) -> None:
    """
    Write predictions to CSV.

    Each line corresponds to an input sequence and contains its predicted item ids
    separated by commas.
    """
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for pred in predictions:
            writer.writerow(pred)

    logger.info(f"Saved predictions to {output_path}")


def main():
    parser = ArgumentParser(description="Predict recommendations with SASRec")
    parser.add_argument(
        "--config",
        type=str,
        default="recsys_project/configs/config_sasrec.py",
        help="Path to python configuration file.",
    )
    parser.add_argument(
        "--config_hyper",
        type=str,
        default="recsys_project/configs/config_training.py",
        help="Path to python configuration file with hyperparameters.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to local checkpoint file.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input data file (CSV with sequences).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save predictions (CSV).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="path"
    )
    args = parser.parse_args()

    # Paths
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    checkpoint_path = Path(args.checkpoint)

    # Load configs
    config: SASRecConfig = load_config(args.config)
    hyper_config: SASRecTrainingConfig = load_config(args.config_hyper)
    logger.info(f"Loaded model config from: {args.config}")
    logger.info(f"Loaded training config from: {args.config_hyper}")

    # Device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Dataset-specific settings
    num_items = get_num_items(config.dataset_name)
    max_seq_len = config.sequence_length - 1  # slice to this length from the end
    top_k = args.top_k
    logger.info(f"Dataset '{config.dataset_name}' has {num_items} items.")
    logger.info(f"max_seq_len={max_seq_len}, top_k={top_k}")

    # Build and load model
    model = build_model(config)
    model.to(device)
    logger.info("Model architecture built and moved to device.")

    load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        map_location=device,
    )

    # Read input CSV into list of sequences
    sequences = read_sequences_from_csv(input_path)

    # Generate top-k predictions
    predictions = generate_topk_predictions(
        model=model,
        sequences=sequences,
        max_seq_len=max_seq_len,
        top_k=top_k,
        device=device,
    )

    # Write predictions to CSV
    write_predictions_to_csv(predictions, output_path)


if __name__ == "__main__":
    main()