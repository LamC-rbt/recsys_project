from argparse import ArgumentParser
import os
import logging
from pathlib import Path
import shutil

import torch

import mlflow

from recsys_project.utils.general_utils import load_config, build_model, get_device, ensure_dir_exists, save_checkpoint
from recsys_project.utils.dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from recsys_project.utils.training_utils import evaluate_and_checkpoint, train_one_epoch
#from torchinfo import summary
from recsys_project.configs.config import SASRecConfig, SASRecTrainingConfig

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
    parser.add_argument('--config_hyper', type=str, default="config_training.py", help="Path to config with training hyperparamters")
    args = parser.parse_args()


    config: SASRecConfig = load_config(args.config)
    hyper_config: SASRecTrainingConfig = load_config(args.config_hyper)
    set_seed(hyper_config.seed)
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
        num_negatives=hyper_config.negs_per_pos,
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

    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("sasrec_recsys")
    run_name = f"sasrec_{config.dataset_name}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("dataset_name", config.dataset_name)
        mlflow.log_param("sequence_length", config.sequence_length)
        mlflow.log_param("embedding_dim", config.embedding_dim)
        mlflow.log_param("dropout_rate", config.dropout_rate)
        mlflow.log_param("num_heads", config.num_heads)
        mlflow.log_param("num_blocks", config.num_blocks)


        mlflow.log_param("train_batch_size", hyper_config.train_batch_size)
        mlflow.log_param("eval_batch_size", hyper_config.eval_batch_size)
        mlflow.log_param("negs_per_pos", hyper_config.negs_per_pos)
        mlflow.log_param("max_epochs", hyper_config.max_epochs)
        mlflow.log_param("max_batches_per_epoch", hyper_config.max_batches_per_epoch)
        mlflow.log_param("val_metric", hyper_config.val_metric)
        mlflow.log_param("early_stopping_patience", hyper_config.early_stopping_patience)
        mlflow.log_param("seed", hyper_config.seed)

        best_metric = float("-inf")
        best_model_path = None
        step = 0
        steps_no_improve = 0

        for epoch in range(hyper_config.max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{hyper_config.max_epochs}")
            avg_loss, full_losses = train_one_epoch(
                model, train_loader, optimizer, device,
                num_items, batches_per_epoch, epoch, logger=logger
            )
            for cur, cur_loss in zip(range(step, step + len(full_losses)), full_losses):
                mlflow.log_metric("train_loss", cur_loss, step=cur)

            step += batches_per_epoch
            mlflow.log_metric("train_epoch_loss", avg_loss, step=epoch)


            best_metric, best_model_path, patience_increase, evaluation_result = evaluate_and_checkpoint(
                model, val_loader, config, hyper_config, device, best_metric, best_model_path, step, logger=logger
            )
            mlflow.log_metrics(
                {str(key).replace("@", ":"): value for key, value in evaluation_result.items()},
                step=(epoch + 1) * batches_per_epoch
            )
            steps_no_improve += patience_increase

            if steps_no_improve >= hyper_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {steps_no_improve} non-improving evaluations.")
                logger.info(f"Best model saved at: {best_model_path}")
                break

        logger.info("Training completed.")
        ensure_dir_exists('hf_checkpoints', logger)
        model.save_pretrained("hf_checkpoints")

        if best_model_path is not None:
            shutil.copy(best_model_path, os.path.join('checkpoints', 'best_model.pt'))
            logger.info("Saved best model")

        if best_model_path is not None and os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

        if os.path.isdir("hf_checkpoints"):
            mlflow.log_artifacts("hf_checkpoints", artifact_path="hf_checkpoints")

        if os.path.exists("dvc.lock"):
            mlflow.log_artifact("dvc.lock")

        if os.path.exists(args.config):
            mlflow.log_artifact(args.config, artifact_path="configs")
        if os.path.exists(args.config_hyper):
            mlflow.log_artifact(args.config_hyper, artifact_path="configs")


if __name__ == "__main__":
    main()