"""Model disagreement analysis for binary classification ensembles.

Measures how often models disagree on predictions, which pairs are most diverse,
and which models catch unique correct predictions that all others miss.
"""

import numpy as np


def build_disagreement_matrix(
    oof_preds: dict[str, np.ndarray],
    labels: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Build asymmetric disagreement matrix.

    Cell (i, j) = number of samples where model i is correct and model j is wrong.

    Args:
        oof_preds: Dict mapping model name -> OOF prediction array.
        labels: Binary ground truth labels (0/1).
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        model_names: Sorted list of model names.
        matrix: Shape (n_models, n_models) integer matrix of disagreement counts.
        correct: Dict mapping model name -> boolean array of per-sample correctness.
    """
    model_names = sorted(oof_preds.keys())
    n = len(model_names)

    correct: dict[str, np.ndarray] = {}
    for name in model_names:
        preds = (oof_preds[name] >= threshold).astype(int)
        correct[name] = preds == labels

    matrix = np.zeros((n, n), dtype=int)
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                continue
            matrix[i, j] = int(np.sum(correct[name_i] & ~correct[name_j]))

    return model_names, matrix, correct


def unique_correct_counts(
    model_names: list[str],
    correct: dict[str, np.ndarray],
) -> dict[str, int]:
    """Count samples where each model is correct when ALL others are wrong.

    Args:
        model_names: List of model names (must match keys in correct).
        correct: Dict mapping model name -> boolean correctness array.

    Returns:
        Dict mapping model name -> count of uniquely correct predictions.
    """
    all_correct = np.column_stack([correct[n] for n in model_names])
    result: dict[str, int] = {}
    for i, name in enumerate(model_names):
        others_correct = np.delete(all_correct, i, axis=1).any(axis=1)
        result[name] = int((correct[name] & ~others_correct).sum())
    return result


def most_diverse_pairs(
    model_names: list[str],
    matrix: np.ndarray,
    top_n: int = 10,
    most_diverse: bool = True,
) -> list[tuple[str, str, int]]:
    """Return top N most (or least) diverse model pairs by symmetric disagreement.

    Args:
        model_names: List of model names.
        matrix: Asymmetric disagreement matrix from build_disagreement_matrix.
        top_n: Number of pairs to return.
        most_diverse: If True, return most diverse pairs; if False, most redundant.

    Returns:
        List of (name_i, name_j, symmetric_disagreement_count) tuples.
    """
    sym = matrix + matrix.T
    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs.append((model_names[i], model_names[j], int(sym[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=most_diverse)
    return pairs[:top_n]


def print_disagreement_report(
    model_names: list[str],
    matrix: np.ndarray,
    correct: dict[str, np.ndarray],
    top_n: int = 10,
) -> None:
    """Print full disagreement report: matrix, symmetric matrix, top pairs, unique correct."""
    short = [n[:12] for n in model_names]
    sym = matrix + matrix.T

    print("\n=== Disagreement Matrix ===")
    print(
        "Cell (row, col) = # samples where ROW model is correct but COL model is wrong\n"
    )
    header = f"{'':>14s}" + "".join(f"{s:>13s}" for s in short)
    print(header)
    print("-" * len(header))
    for i, _name in enumerate(model_names):
        row = f"{short[i]:>14s}"
        for j in range(len(model_names)):
            row += f"{'—':>13s}" if i == j else f"{matrix[i, j]:>13d}"
        print(row)

    print("\n=== Symmetric Disagreement (total unique info per pair) ===")
    print("Higher = more diverse pair (better for ensembling)\n")
    print(header)
    print("-" * len(header))
    for i, _name in enumerate(model_names):
        row = f"{short[i]:>14s}"
        for j in range(len(model_names)):
            row += f"{'—':>13s}" if i == j else f"{sym[i, j]:>13d}"
        print(row)

    print(f"\n=== Top {top_n} Most Diverse Pairs ===")
    for name_i, name_j, count in most_diverse_pairs(model_names, matrix, top_n):
        print(f"  {name_i:35s} vs {name_j:35s}  {count:5d}")

    print(f"\n=== Top {top_n} Most Redundant Pairs ===")
    for name_i, name_j, count in most_diverse_pairs(
        model_names, matrix, top_n, most_diverse=False
    ):
        print(f"  {name_i:35s} vs {name_j:35s}  {count:5d}")

    print("\n=== Unique Correct (right when ALL others are wrong) ===")
    counts = unique_correct_counts(model_names, correct)
    for name in model_names:
        print(f"  {name:35s}  {counts[name]:4d} unique correct")
