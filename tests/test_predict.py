import torch
import pytest

from recsys_project.models.sasrec import SASRec
from recsys_project.utils.prediction_utils import generate_topk_predictions


def test_generate_topk_predictions_basic():
    """Create a small SASRec model and run generate_topk_predictions on a few sequences."""
    device = torch.device("cpu")
    num_items = 10
    sequence_length = 6
    model = SASRec(
        num_items=num_items,
        sequence_length=sequence_length,
        embedding_dim=8,
        num_heads=2,
        num_blocks=1,
        dropout_rate=0.0,
    ).to(device)

    # A couple of example sequences (lists of item IDs)
    sequences = [
        [1, 2, 3],           # short sequence
        [4, 5, 6, 7, 8],     # longer sequence (will be truncated)
        [9],                 # single-item sequence
        [],                  # empty sequence
    ]

    max_seq_len = 4
    top_k = 3

    predictions = generate_topk_predictions(
        model=model,
        sequences=sequences,
        max_seq_len=max_seq_len,
        top_k=top_k,
        device=device,
        logger=None
    )

    # Basic shape checks
    assert isinstance(predictions, list)
    assert len(predictions) == len(sequences)

    for seq, preds in zip(sequences, predictions):
        if not seq:
            # Empty input sequence should give empty predictions
            assert preds == []
            continue

        # Non-empty sequences should produce exactly top_k predictions
        assert isinstance(preds, list)
        assert len(preds) == top_k

        # All predicted items should be valid item IDs and not in the truncated history
        truncated_seq = seq[-max_seq_len:]
        for item_id in preds:
            assert isinstance(item_id, int)
            assert 1 <= item_id <= num_items
            assert item_id not in truncated_seq