from ir_measures import nDCG, R
import json
from pathlib import Path


class SerializableConfig:
    """Base class for configs with JSON serialization (Hugging Face–style)."""

    def to_dict(self):
        def serialize_value(v):
            if isinstance(v, list):
                return [serialize_value(x) for x in v]
            if hasattr(v, "__name__"):
                return v.__name__
            return str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v

        return {k: serialize_value(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{self.__class__.__name__}.json"

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Saved {self.__class__.__name__} to {file_path}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        file_path = Path(load_directory) / f"{cls.__name__}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"No {cls.__name__}.json found in {load_directory}")

        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {cls.__name__} from {file_path}")
        return cls.from_dict(data)

# class SASRecConfig:
#     def __init__(self,
#                  dataset_name,
#                  sequence_length=200,
#                  embedding_dim=256,
#                  train_batch_size=128,
#                  num_heads=4,
#                  num_blocks=3, 
#                  dropout_rate=0.0,
#                  negs_per_pos=256,
#                  max_epochs=10000,
#                  max_batches_per_epoch=100,
#                  metrics=[nDCG@10, R@1, R@10],
#                  val_metric = nDCG@10,
#                  early_stopping_patience=200,
#                  filter_rated=True,
#                  eval_batch_size=512,
#                  recommendation_limit=10,
#                  reuse_item_embeddings=True
#     ):
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.sequence_length = sequence_length
#         self.num_blocks = num_blocks

#         self.dropout_rate = dropout_rate
#         self.negs_per_pos = negs_per_pos
#         self.dataset_name = dataset_name
#         self.train_batch_size = train_batch_size

#         self.max_epochs = max_epochs
#         self.val_metric = val_metric
#         self.max_batches_per_epoch = max_batches_per_epoch
#         self.metrics = metrics

#         self.recommendation_limit = recommendation_limit
#         self.early_stopping_patience = early_stopping_patience
#         self.eval_batch_size = eval_batch_size
#         self.filter_rated = filter_rated
#         self.reuse_item_embeddings = reuse_item_embeddings


class SASRecConfig(SerializableConfig):
    """Configuration class for SASRec model architecture."""

    def __init__(
        self,
        dataset_name: str,
        sequence_length: int = 200,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 3,
        dropout_rate: float = 0.0,
        reuse_item_embeddings: bool = True,
    ):
        self.dataset_name = dataset_name
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.reuse_item_embeddings = reuse_item_embeddings


class SASRecTrainingConfig(SerializableConfig):
    """Configuration class for SASRec training and evaluation hyperparameters."""

    def __init__(
        self,
        train_batch_size: int = 128,
        eval_batch_size: int = 512,
        negs_per_pos: int = 256,
        max_epochs: int = 10000,
        max_batches_per_epoch: int = 100,
        metrics=[nDCG@10, R@1, R@10],
        val_metric = nDCG@10,
        early_stopping_patience: int = 200,
        filter_rated: bool = True,
        recommendation_limit: int = 10,
        seed: int = 42,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.negs_per_pos = negs_per_pos
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.metrics = metrics or []
        self.val_metric = val_metric
        self.early_stopping_patience = early_stopping_patience
        self.filter_rated = filter_rated
        self.recommendation_limit = recommendation_limit
        self.seed = seed