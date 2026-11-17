"""Performance metrics, thresholds and bootstrap utilities."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def recall_at_budget(y, p, budget=0.2):
    n_flag = int(np.ceil(budget * len(y)))
    top = np.argsort(-p, kind="stable")[:n_flag]
    return float(y[top].sum() / max(y.sum(), 1))


def evaluate(y, p, threshold, budget=0.2):
    yhat = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "recall_at_budget": recall_at_budget(y, p, budget),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "flagged": int(yhat.sum()),
        "n": int(len(y)),
    }


def bootstrap_indices(n, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def paired_bootstrap(y, p1, p2, stat, n_boot=1000, seed=42):
    """Bootstrap the difference stat(y, p1) - stat(y, p2) on the same resamples."""
    diffs = []
    for idx in bootstrap_indices(len(y), n_boot, seed):
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(stat(yy, p1[idx]) - stat(yy, p2[idx]))
    diffs = np.array(diffs)
    return {"diff": float(stat(y, p1) - stat(y, p2)), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def bootstrap_ci(y, p, stat, n_boot=1000, seed=42):
    vals = []
    for idx in bootstrap_indices(len(y), n_boot, seed):
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(stat(yy, p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
