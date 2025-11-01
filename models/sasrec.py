import torch
from torch import nn
from .transformer_block import TransformerBlock
from pathlib import Path
import json

class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        sequence_length: int = 200,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 3,
        dropout_rate: float = 0.5,
        reuse_item_embeddings: bool = True,
    ):
        super().__init__()

        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.reuse_item_embeddings = reuse_item_embeddings

        self.embeddings_dropout = nn.Dropout(dropout_rate)
        self.item_embedding = nn.Embedding(self.num_items + 2, self.embedding_dim)  # items enumerated from 1; +1 for padding
        self.position_embedding = nn.Embedding(self.sequence_length, self.embedding_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads,
                             self.embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.seq_norm = nn.LayerNorm(self.embedding_dim)

        if not self.reuse_item_embeddings:
            pass  # add output embeddings later

    def get_output_embeddings(self) -> nn.Embedding:
        return self.item_embedding

    def forward(self, input: torch.Tensor):
        """
        Returns:
            seq_emb: last hidden state of the sequence
            attentions: list of attention weights from each block
        """
        seq = self.item_embedding(input.long())
        mask = (input != self.num_items + 1).float().unsqueeze(-1)

        batch_size = seq.size(0)
        positions = torch.arange(seq.size(1), device=input.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeddings = self.position_embedding(positions)[:input.size(0)]

        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for block in self.transformer_blocks:
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_predictions(self, input: torch.Tensor, limit: int, rated=None):
        with torch.no_grad():
            model_out, _ = self.forward(input)
            seq_emb = model_out[:, -1, :]  # last hidden state

            output_embeddings = self.get_output_embeddings()
            scores = seq_emb @ output_embeddings.weight.T #torch.einsum("bd,nd->bn", seq_emb, output_embeddings.weight)

            # Mask out padding and out-of-range items
            scores[:, 0] = float("-inf")
            scores[:, self.num_items + 1:] = float("-inf")

            # Exclude already rated items
            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float("-inf")

            top_items = torch.topk(scores, limit, dim=1)
            return top_items.indices, top_items.values

    def save_pretrained(self, save_directory: str):
        """Save model weights and config in Hugging Face–compatible format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

        # Save model config
        config_dict = {
            "num_items": self.num_items,
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_blocks": len(self.transformer_blocks),
            "dropout_rate": self.embeddings_dropout.p,
            "reuse_item_embeddings": self.reuse_item_embeddings,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)

        print(f"Model successfully saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model weights and config in Hugging Face–compatible format."""
        load_path = Path(load_directory)
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(load_path / "pytorch_model.bin", map_location="cpu"))
        print(f"Model successfully loaded from {load_directory}")
        return model