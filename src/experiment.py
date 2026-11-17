"""Run all models for one snapshot and cache predictions.

For each week:
  1. build the snapshot, graphs, design matrices and splits;
  2. tune every model on Split B (equal Optuna budget, validation PR-AUC);
  3. retrain every model with 5 seeds on Split B and on Split A using the same
     hyperparameters;
  4. retrain XGBoost and the best GNN without demographic features (Split B);
  5. compute SHAP for the best gradient-boosting model (GNN explanations are computed
     afterwards by explain_gnn.py).
"""
import pickle
import time

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from .data_prep import build_snapshot
from .explain import shap_explain
from .features import design_matrix, make_splits
from .gnn import prepare, run_gnn, train_gnn, _predict
from .graphs import build_graphs
from .tabular import _make, _suggest, run_tabular
from .utils import LOG, ensure_dir

TABULAR = ["LR", "RF", "XGB", "XGB+graph"]
GNNS = ["GCN", "GraphSAGE", "GAT"]
MODELS = TABULAR + GNNS


def _refit_tabular(name, X, y, split, params, cfg):
    from sklearn.preprocessing import StandardScaler
    tr, va, te = split["train"], split["validation"], split["test"]
    Xv = X.to_numpy(np.float32)
    if name == "LR":
        Xv = StandardScaler().fit(Xv[tr]).transform(Xv)
    pw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    p = _suggest(name, optuna.trial.FixedTrial(params))
    out = {"params": params, "val": [], "test": []}
    for seed in cfg["tuning"]["seeds"]:
        m = _make(name, p, seed, pw)
        m.fit(Xv[tr], y[tr])
        out["val"].append(m.predict_proba(Xv[va])[:, 1])
        out["test"].append(m.predict_proba(Xv[te])[:, 1])
    return out


def _refit_gnn(kind, data, params, cfg):
    out = {"params": params, "val": [], "test": [], "models": []}
    for seed in cfg["tuning"]["seeds"]:
        model, _ = train_gnn(kind, data, params, seed, cfg)
        out["val"].append(_predict(model, *data["validation"]))
        out["test"].append(_predict(model, *data["test"]))
        out["models"].append(model)
    return out


def _val_score(res, y_val):
    return float(np.mean([average_precision_score(y_val, p) for p in res["val"]]))


def run_week(tables, week, cfg):
    t0 = time.time()
    cache = ensure_dir(f"{cfg['output_dir']}/cache")
    snap = build_snapshot(tables, week)
    edges, gfeat = build_graphs(snap, tables["studentVle"], week, cfg["graph"]["k_neighbours"])
    X = design_matrix(snap)
    Xg = design_matrix(snap, gfeat)
    Xnd = design_matrix(snap, demographics=False)
    y = snap["at_risk"].to_numpy()
    splits = make_splits(snap, cfg)
    LOG.info("Week %d: %d features (+%d graph features); Split B train/val/test = %d/%d/%d (removed %d overlapping)",
             week, X.shape[1], Xg.shape[1] - X.shape[1], len(splits["B"]["train"]),
             len(splits["B"]["validation"]), len(splits["B"]["test"]), splits["B"]["removed_overlap"])
    res = {"A": {}, "B": {}}
    data = {d: prepare(X, y, edges, splits[d], d) for d in ("A", "B")}

    for name in TABULAR:
        Xm = Xg if name == "XGB+graph" else X
        base = "XGB" if name == "XGB+graph" else name
        tt = time.time()
        res["B"][name] = run_tabular(base, Xm, y, splits["B"], cfg)
        res["A"][name] = _refit_tabular(base, Xm, y, splits["A"], res["B"][name]["params"], cfg)
        LOG.info("  %-10s val PR-AUC %.3f  (%.0f s)", name,
                 _val_score(res["B"][name], y[splits["B"]["validation"]]), time.time() - tt)
    for kind in GNNS:
        tt = time.time()
        res["B"][kind] = run_gnn(kind, data["B"], cfg)
        res["A"][kind] = _refit_gnn(kind, data["A"], res["B"][kind]["params"], cfg)
        LOG.info("  %-10s val PR-AUC %.3f  (%.0f s)", kind,
                 _val_score(res["B"][kind], y[splits["B"]["validation"]]), time.time() - tt)

    yv = y[splits["B"]["validation"]]
    best_gnn = max(GNNS, key=lambda m: _val_score(res["B"][m], yv))
    best_gb = max(["XGB", "XGB+graph"], key=lambda m: _val_score(res["B"][m], yv))
    LOG.info("  best GNN (validation): %s; best gradient boosting: %s", best_gnn, best_gb)

    # ablation without demographic features (same hyperparameters)
    abl = {"XGB": _refit_tabular("XGB", Xnd, y, splits["B"], res["B"]["XGB"]["params"], cfg)}
    data_nd = prepare(Xnd, y, edges, splits["B"], "B")
    abl[best_gnn] = _refit_gnn(best_gnn, data_nd, res["B"][best_gnn]["params"], cfg)
    abl[best_gnn].pop("models")

    # explanations
    expl = {}
    Xgb = Xg if best_gb == "XGB+graph" else X
    gb_model = _make("XGB", _suggest("XGB", optuna.trial.FixedTrial(res["B"][best_gb]["params"])), 0,
                     float((y[splits["B"]["train"]] == 0).sum() / max((y[splits["B"]["train"]] == 1).sum(), 1)))
    gb_model.fit(Xgb.to_numpy(np.float32)[splits["B"]["train"]], y[splits["B"]["train"]])
    expl["shap"] = shap_explain(gb_model, Xgb, splits["B"]["test"])

    for d in res:
        for m in res[d]:
            res[d][m].pop("models", None)
    meta_cols = ["code_module", "code_presentation", "id_student", "at_risk",
                 "grp_gender", "grp_age", "grp_disability", "grp_imd"]
    out = {"week": week, "meta": snap[meta_cols].reset_index(drop=True), "splits": splits, "results": res,
           "best_gnn": best_gnn, "best_gb": best_gb, "ablation": abl, "explain": expl,
           "n_features": X.shape[1], "n_graph_features": Xg.shape[1] - X.shape[1],
           "n_edges": int(len(edges)), "features": list(X.columns), "graph_features": list(gfeat.columns),
           "params": {m: res["B"][m]["params"] for m in MODELS}}
    with open(cache / f"week_{week}.pkl", "wb") as f:
        pickle.dump(out, f)
    LOG.info("Week %d done in %.1f min", week, (time.time() - t0) / 60)
    return out
