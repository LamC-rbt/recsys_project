import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import requests
import json
from pathlib import Path
import numpy as np

URL = "https://raw.githubusercontent.com/kang205/SASRec/refs/heads/master/data/ml-1m.txt"
DATASET_DIR = Path(__file__).parent
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
FILE_PATH = DATASET_DIR / "ml-1m.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def download_dataset(url: str = URL, output_path: Path = FILE_PATH) -> None:
    """
    Downloads the dataset file if not already present.
    """
    if output_path.exists():
        logging.info(f"Dataset already exists at {output_path}")
        return

    logging.info("Downloading dataset...")
    response = requests.get(url, timeout=10, stream=True)
    response.raise_for_status()  # Raises an error for failed requests

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(1024 * 1024):  # 1 MiB chunks
            file.write(chunk)

    logging.info(f"Dataset successfully downloaded to {output_path}")


def dump_statistics(dataset_stats, file_path: Path) -> None:
    with open(file_path, "w") as f:
        json.dump(dataset_stats, f, indent=4)


def read_interactions(file_path: Path, dump: bool = True) -> Dict[int, List[int]]:
    """
    Reads the user-item interactions file and returns a mapping of user -> list of item interactions.
    """
    user_items = defaultdict(list)
    total_interactions = 0
    unique_items: Set[int] = set()

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            if len(parts) != 2:
                continue
            user, item = map(int, parts)
            user_items[user].append(item)
            unique_items.add(item)
            total_interactions += 1

    logging.info(
        "Dataset statistics: users=%d, items=%d, interactions=%d",
        len(user_items),
        len(unique_items),
        total_interactions,
    )
    if dump:
        dataset_stats = {
            "num_users": len(user_items),
            "num_items": len(unique_items), 
            "num_interactions": total_interactions
        }
        dump_statistics(dataset_stats, DATASET_DIR / "dataset_stats.json")

    return user_items


def split_users_for_validation(num_users: int, num_val_users: int = 512, seed: int = 42) -> np.ndarray:
    """
    Randomly selects a subset of users for validation.
    """
    rng = np.random.RandomState(seed)
    val_users = rng.choice(num_users, num_val_users, replace=False)
    return val_users


def prepare_directories() -> None:
    """
    Creates train/val/test directories if they don't already exist.
    """
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        directory.mkdir(exist_ok=True)


def write_sequences(file_path: Path, sequences: List[List[int]]) -> None:
    """
    Writes user interaction sequences to a file, one per line.
    """
    with open(file_path, "w") as f:
        for seq in sequences:
            f.write(" ".join(map(str, seq)) + "\n")


def write_actions(file_path: Path, actions: List[int]) -> None:
    """
    Writes target actions (ground truth) to a file, one per line.
    """
    with open(file_path, "w") as f:
        for action in actions:
            f.write(f"{action}\n")


def split_on_train_val_test() -> None:
    """
    Splits the dataset into train/val/test sets according to the following logic:
    - Last interaction -> test
    - (For 512 selected users) second-last → validation
    """
    prepare_directories()
    user_items = read_interactions(FILE_PATH)
    val_users = split_users_for_validation(num_users=len(user_items))

    train_sequences: List[List[int]] = []
    val_sequences: List[List[int]] = []
    val_targets: List[int] = []
    test_sequences: List[List[int]] = []
    test_targets: List[int] = []

    for user, items in user_items.items():
        if len(items) < 5:
            continue  # skip users with insufficient interactions

        if user in val_users:
            train_sequences.append(items[:-3])
            val_sequences.append(items[:-2])
            val_targets.append(items[-2])
            test_sequences.append(items[:-1])
            test_targets.append(items[-1])
        else:
            train_sequences.append(items[:-2])
            test_sequences.append(items[:-1])
            test_targets.append(items[-1])

    # Write to files
    write_sequences(TRAIN_DIR / "input.txt", train_sequences)
    write_sequences(VAL_DIR / "input.txt", val_sequences)
    write_actions(VAL_DIR / "output.txt", val_targets)
    write_sequences(TEST_DIR / "input.txt", test_sequences)
    write_actions(TEST_DIR / "output.txt", test_targets)

    logging.info("Train, validation, and test files successfully created.")
    logging.info("Train users: %d | Val users: %d | Test users: %d", 
                 len(train_sequences), len(val_sequences), len(test_sequences))


if __name__ == "__main__":
    download_dataset()
    split_on_train_val_test()