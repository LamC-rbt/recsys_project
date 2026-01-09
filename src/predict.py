import logging
from argparse import ArgumentParser
from pathlib import Path


from recsys_project.configs.config import SASRecConfig, SASRecTrainingConfig
from recsys_project.utils.dataset_utils import get_num_items
from recsys_project.utils.general_utils import build_model, get_device, load_config

from recsys_project.utils.prediction_utils import write_predictions_to_csv, generate_topk_predictions, load_model_checkpoint, read_sequences_from_csv


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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

    # Device
    device = get_device()

    # Dataset-specific settings
    num_items = get_num_items(config.dataset_name)
    max_seq_len = config.sequence_length - 1  # slice to this length from the end
    top_k = args.top_k

    # Build and load model
    model = build_model(config)
    model.to(device)

    load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        map_location=device,
        logger=logger
    )

    # Read input CSV into list of sequences
    sequences = read_sequences_from_csv(input_path, logger=logger)

    # Generate top-k predictions
    predictions = generate_topk_predictions(
        model=model,
        sequences=sequences,
        max_seq_len=max_seq_len,
        top_k=top_k,
        device=device,
        logger=logger
    )

    # Write predictions to CSV
    write_predictions_to_csv(predictions, output_path)
    logger.info("Predictions are written successfully")


if __name__ == "__main__":
    main()