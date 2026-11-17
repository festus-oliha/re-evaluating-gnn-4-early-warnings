"""Design matrices and evaluation splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_prep import ASSESSMENT, COUNT_FEATURES, DEMOGRAPHIC_CATS, DEMOGRAPHIC_NUMS, ENGAGEMENT
from .graphs import GRAPH_FEATURES

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]


def design_matrix(snap, graph_feats=None, demographics=True):
    """One-hot encoded feature matrix; counts are log-transformed."""
    X = snap[ENGAGEMENT + ASSESSMENT].astype(float).copy()
    for c in COUNT_FEATURES + ["clicks_last_week", "days_since_last", "n_late"]:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    for m in MODULES:
        X[f"module_{m}"] = (snap["code_module"] == m).astype(float)
    if demographics:
        X["num_of_prev_attempts"] = snap["num_of_prev_attempts"].astype(float)
        X["studied_credits"] = np.log(snap["studied_credits"].astype(float))
        dummies = pd.get_dummies(snap[DEMOGRAPHIC_CATS].astype(str), prefix=DEMOGRAPHIC_CATS, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    if graph_feats is not None:
        X = pd.concat([X, graph_feats.loc[snap.index, GRAPH_FEATURES]], axis=1)
    return X


def make_splits(snap, cfg):
    """Return {'A': {...}, 'B': {...}} with integer row indices for train/validation/test."""
    seed = cfg["seed"]
    y = snap["at_risk"].to_numpy()
    idx = np.arange(len(snap))
    a = cfg["split_a"]
    tr, rest = train_test_split(idx, train_size=a["train"], stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=a["test"] / (a["validation"] + a["test"]),
                              stratify=y[rest], random_state=seed)
    b = cfg["split_b"]
    pres = snap["code_presentation"].to_numpy()
    tr_b = idx[np.isin(pres, b["train"])]
    va_b = idx[np.isin(pres, b["validation"])]
    te_b = idx[np.isin(pres, b["test"])]
    seen = set(snap["id_student"].to_numpy()[tr_b])
    overlap = np.isin(snap["id_student"].to_numpy()[te_b], list(seen))
    te_b = te_b[~overlap]
    return {"A": {"train": tr, "validation": va, "test": te},
            "B": {"train": tr_b, "validation": va_b, "test": te_b, "removed_overlap": int(overlap.sum())}}
