import torch
import pytest
import os

from models.sasrec import SASRec
from config import SASRecConfig


@pytest.mark.parametrize("num_items, batch_size, seq_len, num_blocks", [(3, 2, 5, 1), (11, 7, 3, 2)])
def test_sasrec_forward_pass(num_items, batch_size, seq_len, num_blocks):
    """Ensure SASRec forward pass runs and outputs correct shape."""
    config = SASRecConfig(dataset_name="dummy", sequence_length=seq_len, embedding_dim=8, num_heads=2, num_blocks=num_blocks)
    model = SASRec(
        num_items=num_items, sequence_length=seq_len, embedding_dim=8,
        num_heads=2, num_blocks=num_blocks, dropout_rate=0.5
    )

    input_data = torch.randint(1, num_items+1, (batch_size, seq_len))
    last_hidden_state, attn = model(input_data)

    assert last_hidden_state.shape == (batch_size, seq_len, config.embedding_dim), "Incorrect output shape, should have (batch_size, seq_len, embedding_dim)"
    assert isinstance(attn, list), "Attention weights are stored in an inappropriate structure"
    assert len(attn) == num_blocks
    assert all(isinstance(a, torch.Tensor) for a in attn), "Attention weights for each block must be a torch tensor"
