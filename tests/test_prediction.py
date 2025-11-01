import torch
import pytest
import os

from models.sasrec import SASRec
from config import SASRecConfig


class DummySASRec(SASRec):
    """A subclass of SASRec with a deterministic forward pass for testing."""
    def forward(self, input_ids):
        """
        Returns a tensor shaped (batch_size=1, seq_len, embedding_dim).
        - The last time step is filled with ones (the 'active' embedding).
        - The rest are zeros.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        hidden[:, -1, :] = 1.0  # last position full of ones
        attentions = [torch.zeros(self.num_heads, seq_len, seq_len, device=device)]
        return hidden, attentions


def test_get_predictions_top1():
    """Ensure get_predictions correctly returns top item by similarity using a dummy forward pass."""
    num_items = 10
    config = SASRecConfig(
        dataset_name="dummy",
        sequence_length=3,
        embedding_dim=4,
        num_heads=2,
        num_blocks=1
    )
    
    # Use the dummy subclass instead of the original model
    model = DummySASRec(
        num_items=num_items, sequence_length=3, embedding_dim=4,
        num_heads=2, num_blocks=1
    )

    # Zero out all embeddings, set one special item (id=5) to ones
    with torch.no_grad():
        model.item_embedding.weight.zero_()
        model.item_embedding.weight[5] = 1.0

    # Synthetic input (batch of size 1, arbitrary sequence)
    batch_input = torch.zeros((1, 3), dtype=torch.long)

    # Run get_predictions — should return the item with the all-ones embedding
    top_k_items, scores = model.get_predictions(batch_input, limit=1)

    # Assertions
    assert top_k_items.shape == (1, 1), f"Unexpected shape: {top_k_items.shape}"
    assert top_k_items.item() == 5, f"Expected top item to be 5, got {top_k_items.item()}"
    assert scores.shape == (1, 1), f"Unexpected scores shape: {scores.shape}"