from recsys_project.configs.config import SASRecTrainingConfig


config = SASRecTrainingConfig(
    max_batches_per_epoch=100,
    negs_per_pos=1,
    seed=42
)