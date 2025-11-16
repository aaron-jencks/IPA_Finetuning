#!/usr/bin/env python
import argparse
import os
from multiprocessing import Pool
import pathlib

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


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


def _one_iteration(labels, preds_a, preds_b, metric, observed):
    """
    Perform ONE randomization iteration.
    Returns 1 if |diff| >= |observed|, else 0.
    """
    n = len(labels)
    rng = np.random.default_rng()   # independent RNG per process
    swap = rng.random(n) < 0.5

    a_new = np.where(swap, preds_b, preds_a)
    b_new = np.where(swap, preds_a, preds_b)

    diff = compute_metric(labels, a_new, metric) - \
           compute_metric(labels, b_new, metric)

    return int(abs(diff) >= abs(observed))


def approximate_randomization_test(
    labels,
    preds_a,
    preds_b,
    metric: str = "macro_f1",
    iterations: int = 10000,
):
    """
    Approximate randomization test for multi-class classification.

    - labels: true labels (1D array)
    - preds_a, preds_b: model predictions (1D arrays)
    - metric: 'macro_f1', 'micro_f1', or 'accuracy'
    - iterations: number of randomization iterations
    - seed: optional random seed
    """

    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    observed = compute_metric(labels, preds_a, metric) - compute_metric(
        labels, preds_b, metric
    )

    args = [
        (labels, preds_a, preds_b, metric, observed)
        for _ in range(iterations)
    ]

    count = 0
    with Pool(processes=os.cpu_count()) as pool:
        for r in tqdm(pool.starmap(_one_iteration, args),
                      total=iterations,
                      desc="sampling"):
            count += r

    p_value = (count + 1) / (iterations + 1)
    return observed, p_value


def do_significance_run(ortho, ipa, metric, iterations):
    labels_a, preds_a = load_labels_and_preds(ortho)
    labels_b, preds_b = load_labels_and_preds(ipa)

    # sanity: same labels & order
    check_alignment(labels_a, labels_b)

    labels = labels_a

    # print basic metrics first
    m_a = compute_metric(labels, preds_a, metric)
    m_b = compute_metric(labels, preds_b, metric)

    print(f"Metric: {metric}")
    print(f"Model A (normal): {m_a:.6f}")
    print(f"Model B (ipa): {m_b:.6f}")
    print(f"Observed difference (A - B): {m_a - m_b:.6f}")

    obs_diff, p_value = approximate_randomization_test(
        labels,
        preds_a,
        preds_b,
        metric=metric,
        iterations=iterations,
    )

    print("\nApproximate randomization test")
    print(f"Iterations: {iterations}")
    print(f"Observed diff recomputed: {obs_diff:.6f}")
    print(f"p-value: {p_value:.6g}")

    return m_a, m_b, obs_diff, p_value


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Approximate randomization test for multi-class classification "
            "using predictions saved from HuggingFace Trainer."
        )
    )
    parser.add_argument('logit_path', type=pathlib.Path, help='path to the directory containing logit files')
    parser.add_argument('task', type=str, help='name of the task')
    parser.add_argument('languages', type=str, nargs=2, help='language pair')
    parser.add_argument('date_string', type=str, help='date string')
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

    args = parser.parse_args()

    permutations = [
        ([args.languages[0]], args.languages[0]),
        ([args.languages[0]], args.languages[1]),
        ([args.languages[1]], args.languages[0]),
        ([args.languages[1]], args.languages[1]),
        (args.languages, args.languages[0]),
        (args.languages, args.languages[1]),
    ]

    results = []
    for train_langs, eval_lang in permutations:
        print(f'Testing {train_langs} -> {eval_lang}')
        fname_normal = args.logit_path / f"{args.task}-{'-'.join(train_langs)}-{eval_lang}-normal-{args.date_string}.npz"
        fname_ipa = args.logit_path / f"{args.task}-{'-'.join(train_langs)}-{eval_lang}-ipa-{args.date_string}.npz"
        norm_f1, ipa_f1, _, p_value = do_significance_run(fname_normal, fname_ipa, args.metric, args.iterations)
        results.append((train_langs, eval_lang, norm_f1, ipa_f1, p_value))

    print('\nFinal Results:')
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
