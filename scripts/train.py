from argparse import ArgumentParser
import os
import logging
from pathlib import Path

from config import SASRecConfig, SASRecTrainingConfig

import torch
from utils.general_utils import load_config, build_model, get_device, ensure_dir_exists
from utils.dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from utils.training_utils import evaluate_and_checkpoint, train_one_epoch
#from torchinfo import summary

import random
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = ArgumentParser(description="Train SASRec model.")
    parser.add_argument("--config", type=str, default="config_sasrec.py", help="Path to config file.")
    parser.add_argument('--config_hyper', type)
    args = parser.parse_args()

    set_seed(args.seed)

    config: SASRecConfig = load_config(args.config)
    hyper_config: SASRecTrainingConfig = load_config(args.config_hyper)
    logger.info(f"Loaded configuration: {args.config}")

    ensure_dir_exists("checkpoints", logger)

    num_items = get_num_items(config.dataset_name)
    device = get_device()
    logger.info(f"Using device: {device}")

    model = build_model(config).to(device)
    logger.info("Model successfully built and moved to device.")

    train_loader = get_train_dataloader(
        config.dataset_name,
        batch_size=hyper_config.train_batch_size,
        max_length=config.sequence_length,
        train_neg_per_positive=hyper_config.negs_per_pos,
    )
    val_loader = get_val_dataloader(
        config.dataset_name,
        batch_size=hyper_config.eval_batch_size,
        max_length=config.sequence_length,
    )
    logger.info("Data loaders initialized successfully.")

    optimizer = torch.optim.AdamW(model.parameters())
    batches_per_epoch = min(hyper_config.max_batches_per_epoch, len(train_loader))
    logger.info(f"Training for up to {hyper_config.max_epochs} epochs with {batches_per_epoch} batches per epoch.")

    best_metric = float("-inf")
    best_model_path = None
    step = 0
    steps_no_improve = 0

    for epoch in range(hyper_config.max_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{hyper_config.max_epochs}")
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            num_items, batches_per_epoch, epoch, logger=logger
        )

        step += batches_per_epoch

        best_metric, best_model_path, patience_increase = evaluate_and_checkpoint(
            model, val_loader, config, device, best_metric, best_model_path, step, logger=logger
        )
        steps_no_improve += patience_increase

        if steps_no_improve >= hyper_config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {steps_no_improve} non-improving evaluations.")
            logger.info(f"Best model saved at: {best_model_path}")
            break

    logger.info("Training completed.")
    ensure_dir_exists('hf_checkpoints')
    model.save_pretrained("hf_checkpoints")


if __name__ == "__main__":
    main()