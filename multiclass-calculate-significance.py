#!/usr/bin/env python
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def load_labels_and_preds(path: str):
    """
    Load labels and preds from a .npz file created by your Trainer compute_metrics.

    Expected keys: 'labels', 'preds'.
    """
    data = np.load(path)
    if "labels" not in data or "preds" not in data:
        raise KeyError(f"{path} must contain 'labels' and 'preds' arrays")
    labels = data["labels"]
    preds = data["preds"]
    return labels, preds


def check_alignment(labels_a, labels_b):
    """
    Ensure both models were evaluated on the same examples in the same order.
    """
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            f"Label shapes differ: {labels_a.shape} vs {labels_b.shape}"
        )
    if not np.array_equal(labels_a, labels_b):
        raise ValueError(
            "Label arrays differ between runs. "
            "Make sure both models were evaluated on the exact same eval set "
            "in the same order."
        )


def compute_metric(labels, preds, metric: str):
    """
    Compute the chosen aggregate metric.
    """
    if metric == "macro_f1":
        return f1_score(labels, preds, average="macro")
    elif metric == "micro_f1":
        return f1_score(labels, preds, average="micro")
    elif metric == "accuracy":
        return accuracy_score(labels, preds)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def approximate_randomization_test(
    labels,
    preds_a,
    preds_b,
    metric: str = "macro_f1",
    iterations: int = 10000,
    seed: int | None = None,
):
    """
    Approximate randomization test for multi-class classification.

    - labels: true labels (1D array)
    - preds_a, preds_b: model predictions (1D arrays)
    - metric: 'macro_f1', 'micro_f1', or 'accuracy'
    - iterations: number of randomization iterations
    - seed: optional random seed
    """
    rng = np.random.default_rng(seed)

    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    observed = compute_metric(labels, preds_a, metric) - compute_metric(
        labels, preds_b, metric
    )

    count = 0
    n = len(labels)

    for _ in range(iterations):
        # For each example, swap A/B preds with prob 0.5
        swap = rng.random(n) < 0.5
        a_new = np.where(swap, preds_b, preds_a)
        b_new = np.where(swap, preds_a, preds_b)

        diff = compute_metric(labels, a_new, metric) - compute_metric(
            labels, b_new, metric
        )
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (iterations + 1)
    return observed, p_value


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Approximate randomization test for multi-class classification "
            "using predictions saved from HuggingFace Trainer."
        )
    )
    parser.add_argument(
        "model_a",
        type=str,
        help="Path to .npz file for model A (must contain 'labels' and 'preds')",
    )
    parser.add_argument(
        "model_b",
        type=str,
        help="Path to .npz file for model B (must contain 'labels' and 'preds')",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "micro_f1", "accuracy"],
        help="Metric to test (default: macro_f1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of randomization iterations (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()

    labels_a, preds_a = load_labels_and_preds(args.model_a)
    labels_b, preds_b = load_labels_and_preds(args.model_b)

    # sanity: same labels & order
    check_alignment(labels_a, labels_b)

    labels = labels_a

    # print basic metrics first
    m_a = compute_metric(labels, preds_a, args.metric)
    m_b = compute_metric(labels, preds_b, args.metric)

    print(f"Metric: {args.metric}")
    print(f"Model A ({args.model_a}): {m_a:.6f}")
    print(f"Model B ({args.model_b}): {m_b:.6f}")
    print(f"Observed difference (A - B): {m_a - m_b:.6f}")

    obs_diff, p_value = approximate_randomization_test(
        labels,
        preds_a,
        preds_b,
        metric=args.metric,
        iterations=args.iterations,
        seed=args.seed,
    )

    print("\nApproximate randomization test")
    print(f"Iterations: {args.iterations}")
    print(f"Observed diff recomputed: {obs_diff:.6f}")
    print(f"p-value: {p_value:.6g}")


if __name__ == "__main__":
    main()
