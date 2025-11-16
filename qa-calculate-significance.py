#!/usr/bin/env python
import argparse
import os
import pathlib
import json
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import string

# -------------------------------
# Normalization + SQuAD-style F1
# -------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and whitespace."""
    def remove_articles(text):
        return " ".join(w for w in text.split() if w not in ("a", "an", "the"))

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_squad(prediction: str, ground_truth: str) -> float:
    """
    Standard SQuAD token-overlap F1.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for t in gold_tokens:
        common[t] = common.get(t, 0) + 1

    num_same = 0
    for t in pred_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_ground_truths(prediction, gold_list):
    """
    Handle multiple human answers. Return max F1.
    """
    if not gold_list:
        return 0.0
    return max(f1_score_squad(prediction, gt) for gt in gold_list)


# -------------------------------
# Loading QA JSON (your format)
# -------------------------------

def load_qa_json(path: str):
    """
    Load {"preds": [...], "refs": [...]} file saved during Trainer eval.
    Returns:
      pred_dict: {id -> prediction_text}
      gold_dict: {id -> [gold_answers]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds = obj["preds"]
    refs = obj["refs"]

    pred_dict = {str(p["id"]): p.get("prediction_text", "") for p in preds}

    gold_dict = {}
    for r in refs:
        ans = r["answers"]["text"]
        # always ensure list
        if isinstance(ans, str):
            ans = [ans]
        gold_dict[str(r["id"])] = ans

    return pred_dict, gold_dict


def align_ids(preds_a, golds_a, preds_b, golds_b):
    """
    Ensure we have identical sets of example IDs.
    """
    ids_a = set(preds_a.keys()) & set(golds_a.keys())
    ids_b = set(preds_b.keys()) & set(golds_b.keys())
    common = ids_a & ids_b

    if not common:
        raise ValueError("No common example IDs found between A and B.")

    if ids_a != ids_b:
        raise ValueError("Mismatch in example IDs between predictions A and B.")

    return sorted(common)


# -------------------------------
# Approximate randomization
# -------------------------------

def _one_iteration(scores_a, scores_b, observed):
    """
    Perform one AR iteration: randomly swap scores, return 1 if |diff| >= |observed|.
    """
    rng = np.random.default_rng()
    n = len(scores_a)
    swap = rng.random(n) < 0.5

    a_new = np.where(swap, scores_b, scores_a)
    b_new = np.where(swap, scores_a, scores_b)

    diff = a_new.mean() - b_new.mean()
    return int(abs(diff) >= abs(observed))


def approximate_randomization_test(scores_a, scores_b, iterations):
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    observed = scores_a.mean() - scores_b.mean()

    args = [(scores_a, scores_b, observed) for _ in range(iterations)]

    count = 0
    with Pool(processes=os.cpu_count()) as pool:
        for r in tqdm(pool.starmap(_one_iteration, args),
                      total=iterations,
                      desc="sampling"):
            count += r

    p_value = (count + 1) / (iterations + 1)
    return observed, p_value


# -------------------------------
# QA significance run (like do_significance_run)
# -------------------------------

def do_significance_run(path_normal, path_ipa, iterations):
    preds_a, golds_a = load_qa_json(path_normal)
    preds_b, golds_b = load_qa_json(path_ipa)

    ids = align_ids(preds_a, golds_a, preds_b, golds_b)

    # Compute per-example SQuAD F1
    scores_a = []
    scores_b = []
    for eid in ids:
        gold = golds_a[eid]  # same as golds_b[eid]
        scores_a.append(best_over_ground_truths(preds_a[eid], gold))
        scores_b.append(best_over_ground_truths(preds_b[eid], gold))

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    print(f"Model A (normal) avg F1: {mean_a:.6f}")
    print(f"Model B (ipa)    avg F1: {mean_b:.6f}")
    print(f"Observed difference (A - B): {mean_a - mean_b:.6f}")

    obs_diff, p_value = approximate_randomization_test(scores_a, scores_b, iterations)

    print("\nApproximate randomization test")
    print(f"Iterations: {iterations}")
    print(f"Observed diff recomputed: {obs_diff:.6f}")
    print(f"p-value: {p_value:.6g}")

    return mean_a, mean_b, obs_diff, p_value


# -------------------------------
# Main (mirrors your multiclass)
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Approximate randomization significance test for QA (SQuAD-style)"
    )
    parser.add_argument('logit_path', type=pathlib.Path)
    parser.add_argument('task', type=str)
    parser.add_argument('languages', type=str, nargs=2)
    parser.add_argument('date_string', type=str)
    parser.add_argument('--iterations', type=int, default=10000)

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
        print(f"\nTesting {train_langs} -> {eval_lang}")

        fname_normal = (
            args.logit_path /
            f"{args.task}-{'-'.join(train_langs)}-{eval_lang}-normal-{args.date_string}.json"
        )
        fname_ipa = (
            args.logit_path /
            f"{args.task}-{'-'.join(train_langs)}-{eval_lang}-ipa-{args.date_string}.json"
        )

        f1_norm, f1_ipa, _, p_val = do_significance_run(
            fname_normal, fname_ipa, args.iterations
        )

        results.append((train_langs, eval_lang, f1_norm, f1_ipa, p_val))

    print("\nFinal Results:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
