import torch
from torch import nn

import csv
from pathlib import Path



def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    map_location: torch.device,
    logger=None
) -> None:
    """
    Loads model weights from a local checkpoint file if present.
    """
    if not checkpoint_path.is_file():
        if logger is not None:
            logger.error(f"Checkpoint does not exist: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    state = torch.load(str(checkpoint_path), map_location=map_location)
    model.load_state_dict(state)


def read_sequences_from_csv(input_path: Path, logger=None) -> list[list[int]]:
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
                if logger is not None:
                    logger.warning(f"Line {line_idx} is empty or has no valid integers; skipping.")
                    sequences.append([])
                continue

            try:
                seq = [int(x) for x in tokens]
            except ValueError as e:
                if logger is not None:
                    logger.error(f"Failed to parse integers on line {line_idx}: {row}. Error: {e}")
                raise

            sequences.append(seq)
    return sequences


def generate_topk_predictions(
    model: nn.Module,
    sequences: list[list[int]],
    max_seq_len: int,
    top_k: int,
    device: torch.device,
    logger=None
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
                if logger is not None:
                    logger.warning(f"Sequence {idx+1} is empty; generating empty predictions.")
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
