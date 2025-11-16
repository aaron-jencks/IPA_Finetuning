#!/usr/bin/env python
import argparse
import json
import string
from typing import Dict, List, Tuple

import numpy as np


# -------------------------------
#  SQuAD-style normalization & F1
# -------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        # for English Q/A; adjust if needed
        return " ".join([w for w in text.split() if w.lower() not in ("a", "an", "the")])

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_squad(prediction: str, ground_truth: str) -> float:
    """SQuAD token-overlap F1 between a single prediction and a single gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def em_score_squad(prediction: str, ground_truth: str) -> float:
    """Exact match: 1.0 if normalized strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def best_over_ground_truths(prediction: str, ground_truths: List[str], metric="f1") -> float:
    """Take max F1 (or EM) over multiple gold answers."""
    if metric == "f1":
        scores = [f1_score_squad(prediction, gt) for gt in ground_truths]
    elif metric == "em":
        scores = [em_score_squad(prediction, gt) for gt in ground_truths]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return max(scores) if scores else 0.0


# -------------------------------
#  Loading preds/refs JSON
# -------------------------------

def load_qa_file(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Load a QA preds/refs JSON dumped by make_qa_compute_metrics.

    Expected structure:
      {
        "preds": [{"id": "...", "prediction_text": "..."}],
        "refs":  [{"id": "...", "answers": {"text": [...], "answer_start": [...]}}],
      }

    Returns:
      pred_texts: dict[id -> prediction_text]
      gold_texts: dict[id -> list of gold answers (strings)]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds_list = obj["preds"]
    refs_list = obj["refs"]

    pred_texts = {}
    for p in preds_list:
        pred_texts[str(p["id"])] = p.get("prediction_text", "")

    gold_texts = {}
    for r in refs_list:
        ans = r["answers"]
        texts = ans["text"]
        # ensure list
        if isinstance(texts, str):
            texts = [texts]
        gold_texts[str(r["id"])] = texts

    return pred_texts, gold_texts


def align_ids(
    preds_a: Dict[str, str],
    golds_a: Dict[str, List[str]],
    preds_b: Dict[str, str],
    golds_b: Dict[str, List[str]],
):
    """
    Ensure both models were evaluated on the same example IDs.
    Returns a sorted list of IDs.
    """
    ids_a = set(preds_a.keys()) & set(golds_a.keys())
    ids_b = set(preds_b.keys()) & set(golds_b.keys())
    common = ids_a & ids_b
    if not common:
        raise ValueError("No common IDs found between the two QA outputs.")

    if ids_a != ids_b or ids_a != common:
        # this is stricter than necessary but safer; you can relax if needed
        raise ValueError(
            "Mismatch between example IDs across files. "
            "Make sure you used the same eval set and JSON dumping code."
        )

    return sorted(common)


# -------------------------------
#  Approximate Randomization Test
# -------------------------------

def approximate_randomization_on_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    iterations: int = 10000,
    seed: int | None = None,
) -> Tuple[float, float]:
    """
    Approximate randomization test on per-example scores (F1 or EM).

    scores_a, scores_b: arrays of shape (N,)
    Returns:
      observed_diff (mean(A) - mean(B)), p_value
    """
    rng = np.random.default_rng(seed)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    observed = scores_a.mean() - scores_b.mean()
    n = len(scores_a)
    count = 0

    for _ in range(iterations):
        swap = rng.random(n) < 0.5
        a_new = np.where(swap, scores_b, scores_a)
        b_new = np.where(swap, scores_a, scores_b)
        diff = a_new.mean() - b_new.mean()
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (iterations + 1)
    return observed, p_value


# -------------------------------
#  Main script
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Approximate randomization test for QA (SQuAD-style) using "
            "per-example predictions dumped from HuggingFace Trainer."
        )
    )
    parser.add_argument(
        "qa_a",
        type=str,
        help="Path to JSON file for model A (must contain 'preds' and 'refs').",
    )
    parser.add_argument(
        "qa_b",
        type=str,
        help="Path to JSON file for model B (must contain 'preds' and 'refs').",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "em"],
        help="Metric to test significance on (default: f1).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of randomization iterations (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )

    args = parser.parse_args()

    # Load both models' outputs
    preds_a, golds_a = load_qa_file(args.qa_a)
    preds_b, golds_b = load_qa_file(args.qa_b)

    # Align IDs
    ids = align_ids(preds_a, golds_a, preds_b, golds_b)

    # Compute per-example scores
    scores_a = []
    scores_b = []

    for eid in ids:
        pred_a = preds_a[eid]
        pred_b = preds_b[eid]
        golds = golds_a[eid]  # same as golds_b[eid] by alignment

        score_a = best_over_ground_truths(pred_a, golds, metric=args.metric)
        score_b = best_over_ground_truths(pred_b, golds, metric=args.metric)

        scores_a.append(score_a)
        scores_b.append(score_b)

    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)

    mean_a = scores_a.mean()
    mean_b = scores_b.mean()

    print(f"Metric: {args.metric.upper()}")
    print(f"Model A ({args.qa_a}): {mean_a:.6f}")
    print(f"Model B ({args.qa_b}): {mean_b:.6f}")
    print(f"Observed difference (A - B): {mean_a - mean_b:.6f}")

    observed_diff, p_value = approximate_randomization_on_scores(
        scores_a,
        scores_b,
        iterations=args.iterations,
        seed=args.seed,
    )

    print("\nApproximate randomization test")
    print(f"Iterations: {args.iterations}")
    print(f"Observed diff recomputed: {observed_diff:.6f}")
    print(f"p-value: {p_value:.6g}")


if __name__ == "__main__":
    main()
