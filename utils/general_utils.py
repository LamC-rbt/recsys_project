import importlib
import torch
from config import SASRecConfig, SerializableConfig
from models.sasrec import SASRec
from dataset_utils import get_num_items
import os
from pathlib import Path

import random
import numpy as np

def load_config(config_file: str) -> SerializableConfig:
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.config

def get_device():
    device = "cpu"

    if torch.cuda.is_available():
        device="cuda:0"
    return device


def build_model(config: SASRecConfig):
    num_items = get_num_items(config.dataset_name)

    model = SASRec(
        num_items, sequence_length=config.sequence_length, embedding_dim=config.embedding_dim,
        num_heads=config.num_heads, num_blocks=config.num_blocks, dropout_rate=config.dropout_rate
    )
    return model


def ensure_dir_exists(directory: str, logger) -> None:
    """Ensure that a given directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")


def save_checkpoint(model: torch.nn.Module, path: Path, logger) -> None:
    """Save model state_dict to the specified path."""
    torch.save(model.state_dict(), path)
    logger.info(f"Model checkpoint saved: {path}")


def remove_old_checkpoint(path: Path, logger) -> None:
    """Remove old checkpoint file safely."""
    try:
        if path.exists():
            path.unlink()
            logger.info(f"Removed old checkpoint: {path}")
    except Exception as e:
        logger.warning(f"Failed to remove old checkpoint {path}: {e}")

