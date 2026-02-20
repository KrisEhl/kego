import numpy as np
from sklearn.metrics import roc_auc_score


def hill_climbing(
    oof_matrix: np.ndarray,
    labels: np.ndarray,
    model_names: list[str],
    n_iterations: int = 10,
) -> np.ndarray:
    """Find ensemble weights by greedy hill climbing on AUC."""
    n_models = oof_matrix.shape[1]
    best_weights = np.ones(n_models) / n_models
    best_auc = roc_auc_score(labels, oof_matrix @ best_weights)
    step = 0.01

    for _ in range(n_iterations):
        improved = False
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    continue
                weights = best_weights.copy()
                weights[i] += step
                weights[j] -= step
                if weights[j] < 0:
                    continue
                weights /= weights.sum()
                auc = roc_auc_score(labels, oof_matrix @ weights)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights
                    improved = True
        if not improved:
            break

    return best_weights
