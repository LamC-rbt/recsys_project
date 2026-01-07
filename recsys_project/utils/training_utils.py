import os
from pathlib import Path
import torch
from tqdm import tqdm

from utils.general_utils import remove_old_checkpoint, save_checkpoint
from utils.evaluation_utils import evaluate

def train_one_epoch(model, train_loader, optimizer, device, num_items, batches_per_epoch, epoch_idx, logger):
    """Run one epoch of training and return average loss."""
    model.train()
    batch_iter = iter(train_loader)
    progress_bar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch_idx}")
    total_loss = 0.0

    for batch_idx in progress_bar:
        positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
        model_input = positives[:, :-1]
        labels = positives[:, 1:]
        negatives = negatives[:, 1:, :]

        last_hidden_state, _ = model(model_input)
        output_embeddings = model.get_output_embeddings()
        pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
        pos_neg_embeddings = output_embeddings(pos_neg_concat)

        # Compute logits and ground truth labels
        logits = torch.einsum('bse, bsne -> bsn', last_hidden_state, pos_neg_embeddings)
        ground_truth = torch.zeros_like(logits)
        ground_truth[:, :, 0] = 1  # positive samples in first position

        mask = (model_input != num_items + 1).float()
        loss_per_element = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, ground_truth, reduction="none").mean(-1)
            * mask
        )
        loss = loss_per_element.sum() / mask.sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_description(f"Epoch {epoch_idx} | Loss: {avg_loss:.4f}")

    logger.info(f"Epoch {epoch_idx} training completed. Average loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_and_checkpoint(
    model, val_loader, config, hyper_config, device, best_metric, best_model_path, step, logger
):
    """Evaluate model and optionally save checkpoint if improved."""
    logger.info("Evaluating model on validation set...")
    evaluation_result = evaluate(
        model,
        val_loader,
        hyper_config.metrics,
        hyper_config.recommendation_limit,
        hyper_config.filter_rated,
        device=device,
    )

    metric_value = evaluation_result[hyper_config.val_metric]
    logger.info(f"Validation result ({hyper_config.val_metric}): {metric_value:.6f}")

    if metric_value > best_metric:
        logger.info(f"Validation metric improved: {best_metric:.6f} → {metric_value:.6f}")
        model_name = (
            f"checkpoints/gsasrec-{config.dataset_name}-"
            f"step:{step}-negs:{hyper_config.negs_per_pos}-"
            f"emb:{config.embedding_dim}-dropout:{config.dropout_rate}-"
            f"metric:{metric_value:.6f}.pt"
        )

        new_checkpoint_path = Path(model_name)
        if best_model_path is not None:
            remove_old_checkpoint(Path(best_model_path), logger)
        save_checkpoint(model, new_checkpoint_path, logger)

        return metric_value, str(new_checkpoint_path), 0  # reset patience
    else:
        logger.info("Validation metric did not improve.")
        return best_metric, best_model_path, 1
