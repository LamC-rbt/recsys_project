import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for sequential recommendation tasks.
    Each example represents a user sequence of item interactions.

    Args:
        input_file (str | Path): Path to the file containing item sequences.
        padding_value (int): Value used for sequence padding.
        output_file (Optional[str | Path]): Path to the file containing target items (optional).
        max_length (int): Maximum sequence length to consider.
    """

    def __init__(
        self,
        input_file: str,
        padding_value: int,
        output_file: Optional[str] = None,
        max_length: int = 200,
    ):
        input_file = Path(input_file)
        self.max_length = max_length
        self.padding_value = padding_value

        # Load input sequences
        with input_file.open("r") as f:
            self.inputs: List[List[int]] = [
                list(map(int, line.strip().split())) for line in f if line.strip()
            ]

        # Load targets if available
        if output_file:
            with open(output_file, "r") as f:
                self.outputs: Optional[List[int]] = [
                    int(line.strip()) for line in f if line.strip()
                ]
        else:
            self.outputs = None

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple:
        sequence = self.inputs[idx]
        rated_items: Set[int] = set(sequence)

        # Truncate or pad sequence to max length
        if len(sequence) > self.max_length:
            sequence = sequence[-self.max_length:]
        elif len(sequence) < self.max_length:
            padding = [self.padding_value] * (self.max_length - len(sequence))
            sequence = padding + sequence

        sequence_tensor = torch.tensor(sequence, dtype=torch.long)

        if self.outputs is not None:
            target_tensor = torch.tensor(self.outputs[idx], dtype=torch.long)
            return sequence_tensor, rated_items, target_tensor

        return (sequence_tensor,)



def collate_with_random_negatives(
    batch: List[Tuple[torch.Tensor]],
    pad_value: int,
    num_negatives: int,
) -> List[torch.Tensor]:
    """
    Collate function for training.
    Generates random negative samples for each sequence.
    """
    input_batch = torch.stack([item[0] for item in batch], dim=0)
    negatives = torch.randint(
        low=1,
        high=pad_value,
        size=(input_batch.size(0), input_batch.size(1), num_negatives),
    )
    return [input_batch, negatives]


def collate_val_test(batch: List[Tuple[torch.Tensor, Set[int], torch.Tensor]]) -> List:
    """
    Collate function for validation and test datasets.
    Returns sequences, rated items, and ground-truth targets.
    """
    inputs = torch.stack([item[0] for item in batch], dim=0)
    rated_sets = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch], dim=0)
    return [inputs, rated_sets, targets]



def load_dataset_stats(dataset_dir: Path) -> dict:
    """Loads dataset statistics from the dataset_stats.json file."""
    stats_path = Path(dataset_dir) / "dataset_stats.json"
    with stats_path.open("r") as f:
        return json.load(f)


def get_num_items(dataset_name: str) -> int:
    """Returns the number of items in the dataset."""
    stats = load_dataset_stats(Path(f"datasets/{dataset_name}"))
    return stats["num_items"]


def get_padding_value(dataset_dir: str) -> int:
    """Returns the padding value (num_items + 1) for the dataset."""
    stats = load_dataset_stats(Path(dataset_dir))
    return stats["num_items"] + 1


def get_train_dataloader(
    dataset_name: str,
    batch_size: int = 32,
    max_length: int = 200,
    num_negatives: int = 256,
) -> DataLoader:
    """
    Creates a DataLoader for the training set.

    Args:
        dataset_name (str): Name of the dataset (inside 'datasets/').
        batch_size (int): Number of sequences per batch.
        max_length (int): Maximum sequence length (+1 for shifted training).
        num_negatives (int): Number of negative samples per positive.

    Returns:
        DataLoader: Training dataloader with random negative sampling.
    """
    dataset_dir = Path(f"datasets/{dataset_name}")
    padding_value = get_padding_value(dataset_dir)

    dataset = SequenceDataset(
        input_file=dataset_dir / "train" / "input.txt",
        padding_value=padding_value,
        max_length=max_length + 1,  # +1 for sequence shifting
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_with_random_negatives(
            batch, padding_value, num_negatives
        ),
    )


def get_val_or_test_dataloader(
    dataset_name: str,
    split: str = "val",
    batch_size: int = 32,
    max_length: int = 200,
) -> DataLoader:
    """
    Creates a DataLoader for validation or test sets.

    Args:
        dataset_name (str): Name of the dataset (inside 'datasets/').
        split (str): Either 'val' or 'test'.
        batch_size (int): Number of sequences per batch.
        max_length (int): Maximum sequence length.

    Returns:
        DataLoader: Validation or test dataloader.
    """
    dataset_dir = Path(f"datasets/{dataset_name}")
    padding_value = get_padding_value(dataset_dir)

    dataset = SequenceDataset(
        input_file=dataset_dir / split / "input.txt",
        output_file=dataset_dir / split / "output.txt",
        padding_value=padding_value,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_val_test,
    )


def get_val_dataloader(dataset_name: str, batch_size: int = 32, max_length: int = 200) -> DataLoader:
    """Wrapper for creating the validation dataloader."""
    return get_val_or_test_dataloader(dataset_name, "val", batch_size, max_length)


def get_test_dataloader(dataset_name: str, batch_size: int = 32, max_length: int = 200) -> DataLoader:
    """Wrapper for creating the test dataloader."""
    return get_val_or_test_dataloader(dataset_name, "test", batch_size, max_length)