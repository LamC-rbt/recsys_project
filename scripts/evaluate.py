import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from config import SASRecConfig, SASRecTrainingConfig

import torch
from torch import nn

from utils.dataset_utils import get_num_items, get_test_dataloader
from utils.evaluation_utils import evaluate
from utils.general_utils import build_model, get_device, load_config


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    hub_model_name: str = None,
    map_location: torch.device = None
) -> None:
    """
    Loads model weights from a local checkpoint file if present, otherwise
    optionally downloads from Hugging Face Hub if hub_model_name is provided.
    Logs progress and errors.

    Args:
        model: The nn.Module model to load weights into.
        checkpoint_path: Path to local checkpoint file.
        hub_model_name: Name of model on Hugging Face Hub (optional).
        map_location: map_location argument for torch.load.
    """
    if checkpoint_path.is_file():
        logger.info(f"Loading model state dict from local checkpoint: {checkpoint_path}")
        state = torch.load(str(checkpoint_path), map_location=map_location)
        model.load_state_dict(state)
        logger.info("Successfully loaded local checkpoint.")
    # elif hub_model_name:
    #     logger.info(f"No local checkpoint found at {checkpoint_path}. Downloading from Hugging Face Hub: '{hub_model_name}'")
    #     # Example: if your model supports a from_pretrained, you might do:
    #     # model = ModelClass.from_pretrained(hub_model_name)
    #     # But assuming build_model and state dict approach:
    #     from huggingface_hub import hf_hub_download  # optional dependency
    #     try:
    #         # Download the file from HF
    #         hf_file = hf_hub_download(repo_id=hub_model_name, filename=checkpoint_path.name)
    #         logger.info(f"Downloaded hub checkpoint to {hf_file}")
    #         state = torch.load(hf_file, map_location=map_location)
    #         model.load_state_dict(state)
    #         logger.info("Successfully loaded weights from Hugging Face Hub.")
    #     except Exception as e:
    #         logger.error(f"Failed to download or load weights from hub: {e}")
    #         raise
    else:
        logger.error(f"Neither local checkpoint found at {checkpoint_path} nor hub model name provided.")
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")


def main():
    parser = ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to python configuration file.")
    parser.add_argument('--config_hyper', type=str, required=True,
                        help="Path to python configuration file with hyperparameters.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to local checkpoint file.")
    parser.add_argument('--hub_model', type=str, default=None,
                        help="Hugging Face Hub model identifier (optional).")
    args = parser.parse_args()

    # Load config
    config: SASRecConfig = load_config(args.config)
    hyper_config: SASRecTrainingConfig = load_config(args.config_hyper)
    logger.info(f"Loaded configuration from: {args.config}")

    # Determine device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Get number of items (dataset specific)
    num_items = get_num_items(config.dataset_name)
    logger.info(f"Dataset '{config.dataset_name}' has {num_items} items.")

    # Build model
    model = build_model(config)
    model = model.to(device)
    logger.info("Model architecture built and moved to device.")

    # Load weights
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path if checkpoint_path else Path(""),
        hub_model_name=args.hub_model,
        map_location=device
    )

    # Prepare test dataloader
    test_loader = get_test_dataloader(
        config.dataset_name,
        batch_size=hyper_config.eval_batch_size,
        max_length=config.sequence_length
    )
    logger.info(f"Test DataLoader prepared: batch_size={hyper_config.eval_batch_size}, max_length={config.sequence_length}")

    # Evaluate
    logger.info("Starting evaluation...")
    evaluation_result = evaluate(
        model=model,
        data_loader=test_loader,
        metrics=hyper_config.metrics,
        top_k=hyper_config.recommendation_limit,
        filter_rated=hyper_config.filter_rated,
        device=device
    )
    logger.info(f"Evaluation results: {evaluation_result}")

    # Print or return result
    # print(evaluation_result)

if __name__ == "__main__":
    main()