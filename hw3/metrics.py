import numpy as np

def recall_topk(ranked_indices, true_indices, k=5):
    topk = ranked_indices[:, :k]
    hits = (topk == true_indices[:, None]).any(axis=1)
    return np.mean(hits)

def mean_reciprocal_rank(ranked_indices, true_indices):
    reciprocal_ranks = []
    for preds, true in zip(ranked_indices, true_indices):
        pos = np.where(preds == true)[0]
        if pos.size:
            reciprocal_ranks.append(1.0 / (pos[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)