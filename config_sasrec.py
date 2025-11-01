from config import SASRecConfig

config = SASRecConfig(
    dataset_name='MovieLens_Large',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    num_blocks=2,
    dropout_rate=0.5,
    reuse_item_embeddings=True
)