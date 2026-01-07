import torch
import tqdm
from models.sasrec import SASRec
from ir_measures import ScoredDoc, Qrel
import ir_measures


def evaluate(
    model: SASRec,
    data_loader,
    metrics,
    top_k: int,
    filter_rated: bool,
    device: torch.device,
):
    """
    Evaluate a SASRec model using IR metrics such as NDCG, Recall, Precision, etc.

    Args:
        model (SASRec): Trained SASRec model.
        data_loader (DataLoader): Iterable yielding batches of (sequence, rated_items, target_items).
        metrics (list): List of IR metrics from `ir_measures` (e.g., [ir_measures.nDCG@10]).
        top_k (int): Number of top recommendations to consider.
        filter_rated (bool): Whether to exclude already-rated items from recommendations.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').

    Returns:
        dict: Aggregated evaluation results (e.g., {'nDCG@10': 0.321, 'Recall@10': 0.512})
    """
    model.eval()
    user_index = 0
    scored_documents = []
    relevance_labels = []

    with torch.no_grad():
        total_batches = len(data_loader)
        progress_bar = tqdm.tqdm(enumerate(data_loader), total=total_batches, desc="Evaluating")

        for batch_idx, (sequences, rated_items, targets) in progress_bar:
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Get model predictions
            if filter_rated:
                recommended_items, predicted_scores = model.get_predictions(sequences, top_k, rated_items)
            else:
                recommended_items, predicted_scores = model.get_predictions(sequences, top_k)

            # Collect scored documents and relevance judgments
            for rec_items, rec_scores, true_item in zip(recommended_items, predicted_scores, targets):
                user_id = str(user_index)

                # Record model scores
                for item_id, score in zip(rec_items, rec_scores):
                    scored_documents.append(ScoredDoc(user_id, str(item_id.item()), score.item()))

                # Record ground truth relevance
                relevance_labels.append(Qrel(user_id, str(true_item.item()), 1))

                user_index += 1

    # Compute aggregate IR metrics
    results = ir_measures.calc_aggregate(metrics, relevance_labels, scored_documents)
    return results
