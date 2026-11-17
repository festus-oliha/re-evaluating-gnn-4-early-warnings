"""Tables, hypothesis tests, fairness analysis and figures from cached predictions."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score

from .experiment import GNNS, MODELS, TABULAR
from .figures import make_figures
from .metrics import best_f1_threshold, bootstrap_ci, bootstrap_indices, evaluate, paired_bootstrap
from .utils import LOG, ensure_dir, save_json

ATTRS = ["gender", "age", "disability", "imd"]


def load_cache(cfg):
    out = {}
    for f in sorted(Path(cfg["output_dir"], "cache").glob("week_*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        out[d["week"]] = d
    return dict(sorted(out.items()))


def _parts(d, design):
    sp = d["splits"][design]
    y = d["meta"]["at_risk"].to_numpy()
    return sp, y[sp["validation"]], y[sp["test"]]


def _ensemble(res):
    return np.mean(res["val"], axis=0), np.mean(res["test"], axis=0)


def _rates(y, yhat, mask):
    pos, neg = (y == 1) & mask, (y == 0) & mask
    tpr = yhat[pos].mean() if pos.any() else np.nan
    fpr = yhat[neg].mean() if neg.any() else np.nan
    sel = yhat[mask].mean() if mask.any() else np.nan
    return tpr, fpr, sel


def fairness(y, yhat, groups):
    """EOD, equal opportunity difference and DPD for a binary interest/reference grouping."""
    ok = pd.notna(groups)
    y, yhat, g = y[ok], yhat[ok], np.asarray(groups)[ok]
    ti, fi, si = _rates(y, yhat, g == "interest")
    tr, fr, sr = _rates(y, yhat, g == "reference")
    return {"eod": max(abs(ti - tr), abs(fi - fr)), "eopd": abs(ti - tr), "dpd": abs(si - sr),
            "recall_interest": ti, "recall_reference": tr, "fpr_interest": fi, "fpr_reference": fr}


def performance_tables(data, cfg):
    budget = cfg["evaluation"]["alert_budget"]
    seed_rows, ens_rows = [], []
    for week, d in data.items():
        for design in ("A", "B"):
            sp, yv, yt = _parts(d, design)
            for m in MODELS:
                r = d["results"][design][m]
                for s, (pv, pt) in enumerate(zip(r["val"], r["test"])):
                    met = evaluate(yt, pt, best_f1_threshold(yv, pv), budget)
                    seed_rows.append({"week": week, "split": design, "model": m, "seed": s,
                                      "val_pr_auc": average_precision_score(yv, pv), **met})
                pv, pt = _ensemble(r)
                met = evaluate(yt, pt, best_f1_threshold(yv, pv), budget)
                lo, hi = bootstrap_ci(yt, pt, average_precision_score, cfg["evaluation"]["n_bootstrap"])
                ens_rows.append({"week": week, "split": design, "model": m, **met,
                                 "pr_auc_ci_low": lo, "pr_auc_ci_high": hi})
    seeds = pd.DataFrame(seed_rows)
    metrics = ["val_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_budget", "brier"]
    summ = seeds.groupby(["split", "week", "model"], sort=False)[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    return seeds, summ.reset_index(), pd.DataFrame(ens_rows)


def _best(d, design, candidates):
    sp, yv, _ = _parts(d, design)
    return max(candidates, key=lambda m: np.mean([average_precision_score(yv, p) for p in d["results"][design][m]["val"]]))


def hypothesis_tests(data, cfg):
    nb = cfg["evaluation"]["n_bootstrap"]
    margin = cfg["evaluation"]["h1_margin"]
    ap = average_precision_score
    rq1, h1, h2 = [], [], []
    for week, d in data.items():
        sp, yv, yt = _parts(d, "B")
        g = d["best_gnn"]
        pg = _ensemble(d["results"]["B"][g])[1]
        for base in ("XGB", "XGB+graph"):
            pb = _ensemble(d["results"]["B"][base])[1]
            r = paired_bootstrap(yt, pg, pb, ap, nb)
            row = {"week": week, "best_gnn": g, "baseline": base, "gnn_pr_auc": ap(yt, pg),
                   "baseline_pr_auc": ap(yt, pb), **r}
            rq1.append(row)
            if base == "XGB+graph":
                match = (r["ci_low"] >= -margin and r["ci_high"] <= margin) or r["ci_high"] <= 0
                h1.append({**row, "matches_within_margin": bool(match)})
        for design in ("A", "B"):
            _, yv2, yt2 = _parts(d, design)
            bg = _best(d, design, GNNS)
            bt = _best(d, design, TABULAR)
            p1, p2 = _ensemble(d["results"][design][bg])[1], _ensemble(d["results"][design][bt])[1]
            r = paired_bootstrap(yt2, p1, p2, ap, nb)
            px = _ensemble(d["results"][design]["XGB"])[1]
            rx = paired_bootstrap(yt2, p1, px, ap, nb)
            h2.append({"week": week, "split": design, "best_gnn": bg, "best_tabular": bt,
                       "gnn_pr_auc": ap(yt2, p1), "tabular_pr_auc": ap(yt2, p2),
                       "gnn_advantage": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                       "xgb_pr_auc": ap(yt2, px), "gnn_minus_xgb": rx["diff"],
                       "gnn_minus_xgb_ci_low": rx["ci_low"], "gnn_minus_xgb_ci_high": rx["ci_high"]})
    rq1, h1, h2 = pd.DataFrame(rq1), pd.DataFrame(h1), pd.DataFrame(h2)
    piv = h2.pivot(index="week", columns="split", values="gnn_advantage")
    piv["shrinks_under_B"] = piv["B"] < piv["A"]
    verdict = {
        "H1": {"weeks_matching": int(h1["matches_within_margin"].sum()), "n_weeks": len(h1),
               "supported": bool(h1["matches_within_margin"].sum() >= np.ceil(0.6 * len(h1)))},
        "H2": {"weeks_shrinking": int(piv["shrinks_under_B"].sum()), "n_weeks": len(piv),
               "supported": bool(piv["shrinks_under_B"].sum() >= np.ceil(0.6 * len(piv)))},
    }
    return rq1, h1, h2, piv.reset_index(), verdict


def split_drop(summ):
    p = summ.pivot_table(index=["week", "model"], columns="split", values="pr_auc_mean").reset_index()
    p["drop_A_to_B"] = p["A"] - p["B"]
    return p


def fairness_analysis(data, cfg):
    nb = min(cfg["evaluation"]["n_bootstrap"], 1000)
    rows, h3, mit = [], [], []
    for week, d in data.items():
        sp, yv, yt = _parts(d, "B")
        meta_v = d["meta"].iloc[sp["validation"]]
        meta_t = d["meta"].iloc[sp["test"]]
        preds = {}
        for role, m in (("best_gnn", d["best_gnn"]), ("best_gb", d["best_gb"])):
            pv, pt = _ensemble(d["results"]["B"][m])
            thr = best_f1_threshold(yv, pv)
            preds[role] = (m, pv, pt, thr, (pt >= thr).astype(int))
        for attr in ATTRS:
            g = meta_t[f"grp_{attr}"].to_numpy(dtype=object)
            for role, (m, pv, pt, thr, yhat) in preds.items():
                f = fairness(yt, yhat, g)
                ok = pd.notna(g)
                f["eod_fairlearn"] = equalized_odds_difference(yt[ok], yhat[ok], sensitive_features=g[ok])
                f["dpd_fairlearn"] = demographic_parity_difference(yt[ok], yhat[ok], sensitive_features=g[ok])
                rows.append({"week": week, "attribute": attr, "role": role, "model": m, **f})
            # H3: paired bootstrap of EOD difference (GNN - GB)
            yg, yb = preds["best_gnn"][4], preds["best_gb"][4]
            diffs = []
            for idx in bootstrap_indices(len(yt), nb):
                diffs.append(fairness(yt[idx], yg[idx], g[idx])["eod"] - fairness(yt[idx], yb[idx], g[idx])["eod"])
            diffs = np.array(diffs)
            point = fairness(yt, yg, g)["eod"] - fairness(yt, yb, g)["eod"]
            h3.append({"week": week, "attribute": attr, "best_gnn": preds["best_gnn"][0],
                       "best_gb": preds["best_gb"][0], "eod_gnn": fairness(yt, yg, g)["eod"],
                       "eod_gb": fairness(yt, yb, g)["eod"], "diff": point,
                       "ci_low": float(np.nanpercentile(diffs, 2.5)), "ci_high": float(np.nanpercentile(diffs, 97.5)),
                       "gnn_significantly_fairer": bool(np.nanpercentile(diffs, 97.5) < 0)})
            # mitigation: group-specific thresholds (equalized odds), fitted on validation
            gv = meta_v[f"grp_{attr}"].to_numpy(dtype=object)
            okv, okt = pd.notna(gv), pd.notna(g)
            for role, (m, pv, pt, thr, yhat) in preds.items():
                to = ThresholdOptimizer(estimator=_ScoreModel().fit(None, None), constraints="equalized_odds",
                                        objective="balanced_accuracy_score", prefit=True,
                                        predict_method="predict_proba")
                to.fit(pv[okv].astype(np.float64).reshape(-1, 1), yv[okv], sensitive_features=gv[okv])
                ym = to.predict(pt[okt].astype(np.float64).reshape(-1, 1), sensitive_features=g[okt], random_state=cfg["seed"])
                before = fairness(yt[okt], yhat[okt], g[okt])
                after = fairness(yt[okt], np.asarray(ym), g[okt])
                ev_b = evaluate(yt[okt], pt[okt], thr)
                rec_a = float(np.asarray(ym)[yt[okt] == 1].mean())
                prec_a = float(yt[okt][np.asarray(ym) == 1].mean()) if np.asarray(ym).sum() else 0.0
                f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-12)
                mit.append({"week": week, "attribute": attr, "role": role, "model": m,
                            "eod_before": before["eod"], "eod_after": after["eod"],
                            "recall_before": ev_b["recall"], "recall_after": rec_a,
                            "f1_before": ev_b["f1"], "f1_after": f1_a})
    h3 = pd.DataFrame(h3)
    verdict = {"H3": {"comparisons": len(h3), "gnn_significantly_fairer": int(h3["gnn_significantly_fairer"].sum()),
                      "supported": bool(h3["gnn_significantly_fairer"].sum() <= len(h3) / 2)}}
    return pd.DataFrame(rows), h3, pd.DataFrame(mit), verdict


class _ScoreModel(ClassifierMixin, BaseEstimator):
    """Pre-fitted 'model' whose input is already a risk score (for ThresholdOptimizer)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (np.asarray(X).ravel() >= 0.5).astype(int)


def extra_tables(data, cfg):
    """Split B without module CCC; ablation without demographics; explanations; data summary."""
    noccc, abl, shap_rows, gx_rows, summary, reliance, perm_rows = [], [], [], [], [], [], []
    for week, d in data.items():
        meta = d["meta"]
        sp, yv, yt = _parts(d, "B")
        mod_t = meta["code_module"].to_numpy()[sp["test"]]
        keep = mod_t != "CCC"
        for m in MODELS:
            pv, pt = _ensemble(d["results"]["B"][m])
            thr = best_f1_threshold(yv, pv)
            noccc.append({"week": week, "model": m, "pr_auc_all": average_precision_score(yt, pt),
                          "pr_auc_without_ccc": average_precision_score(yt[keep], pt[keep]),
                          "recall_without_ccc": evaluate(yt[keep], pt[keep], thr)["recall"]})
        for m, r in d["ablation"].items():
            pv, pt = _ensemble(r)
            pv0, pt0 = _ensemble(d["results"]["B"][m])
            g = meta.iloc[sp["test"]]
            row = {"week": week, "model": m, "pr_auc_with_demographics": average_precision_score(yt, pt0),
                   "pr_auc_without_demographics": average_precision_score(yt, pt)}
            for attr in ATTRS:
                gg = g[f"grp_{attr}"].to_numpy(dtype=object)
                row[f"eod_{attr}_with"] = fairness(yt, (pt0 >= best_f1_threshold(yv, pv0)).astype(int), gg)["eod"]
                row[f"eod_{attr}_without"] = fairness(yt, (pt >= best_f1_threshold(yv, pv)).astype(int), gg)["eod"]
            abl.append(row)
        imp = d["explain"]["shap"]
        for rank, (f, v) in enumerate(imp.head(15).items(), 1):
            shap_rows.append({"week": week, "model": d["best_gb"], "rank": rank, "feature": f, "mean_abs_shap": v})
        if "gnnexplainer" in d["explain"] and "neighbour_share" not in d["explain"]["gnnexplainer"]:
            gx = d["explain"]["gnnexplainer"]
            for rank, (f, v) in enumerate(gx["feature_importance"].head(15).items(), 1):
                gx_rows.append({"week": week, "model": d["best_gnn"], "rank": rank, "feature": f,
                                "importance": v, "n_students": gx["n_students"]})
        if "gnn_permutation" in d["explain"]:
            for rank, (f, v) in enumerate(d["explain"]["gnn_permutation"].head(15).items(), 1):
                perm_rows.append({"week": week, "model": d["best_gnn"], "rank": rank, "feature": f, "pr_auc_drop": v})
        if "graph_reliance" in d["explain"]:
            reliance.append({"week": week, **d["explain"]["graph_reliance"]})
        y = meta["at_risk"].to_numpy()
        for design in ("A", "B"):
            s = d["splits"][design]
            summary.append({"week": week, "split": design, "registrations": len(meta),
                            "at_risk_rate": y.mean(), "train": len(s["train"]), "validation": len(s["validation"]),
                            "test": len(s["test"]), "test_at_risk_rate": y[s["test"]].mean(),
                            "removed_overlap": s.get("removed_overlap", 0), "features": d["n_features"],
                            "graph_features": d["n_graph_features"], "edges": d["n_edges"],
                            "best_gnn": d["best_gnn"], "best_gb": d["best_gb"]})
    return (pd.DataFrame(noccc), pd.DataFrame(abl), pd.DataFrame(shap_rows), pd.DataFrame(gx_rows),
            pd.DataFrame(summary), pd.DataFrame(reliance), pd.DataFrame(perm_rows))


def run_analysis(cfg):
    data = load_cache(cfg)
    if not data:
        LOG.info("No cached results found in %s/cache", cfg["output_dir"])
        return
    t = ensure_dir(Path(cfg["output_dir"]) / "tables")
    seeds, summ, ens = performance_tables(data, cfg)
    rq1, h1, h2, h2piv, verdict = hypothesis_tests(data, cfg)
    drop = split_drop(summ)
    fair, h3, mit, v3 = fairness_analysis(data, cfg)
    verdict.update(v3)
    noccc, abl, shap_t, gx_t, summary, reliance, perm = extra_tables(data, cfg)
    params = pd.DataFrame([{"week": w, "model": m, **p} for w, d in data.items() for m, p in d["params"].items()])
    out = {"t01_data_summary": summary, "t02_performance_seeds": seeds, "t03_performance_summary": summ,
           "t04_performance_ensemble": ens, "t05_rq1_gnn_vs_xgb": rq1, "t06_h1": h1, "t07_h2_advantage": h2,
           "t08_h2_pivot": h2piv, "t09_split_drop": drop, "t10_fairness": fair, "t11_h3": h3,
           "t12_mitigation": mit, "t13_without_ccc": noccc, "t14_ablation_demographics": abl,
           "t15_shap": shap_t, "t16_gnnexplainer": gx_t, "t17_hyperparameters": params,
           "t18_graph_reliance": reliance, "t19_gnn_permutation": perm}
    for name, df in out.items():
        df.to_csv(t / f"{name}.csv", index=False)
    save_json(verdict, Path(cfg["output_dir"]) / "hypotheses.json")
    make_figures(out, cfg)
    LOG.info("Hypotheses: %s", verdict)
    _summary_md(out, verdict, cfg)


def _summary_md(out, verdict, cfg):
    s = out["t03_performance_summary"]
    lines = ["# Results summary", "", "## Test PR-AUC (mean ± sd over seeds)", ""]
    for split in ("B", "A"):
        p = s[s["split"] == split].pivot(index="model", columns="week", values="pr_auc_mean")
        sd = s[s["split"] == split].pivot(index="model", columns="week", values="pr_auc_std")
        lines.append(f"### Split {split}")
        lines.append("| Model | " + " | ".join(f"Week {w}" for w in p.columns) + " |")
        lines.append("|---" * (len(p.columns) + 1) + "|")
        for m in MODELS:
            if m in p.index:
                lines.append(f"| {m} | " + " | ".join(f"{p.loc[m, w]:.3f} ± {sd.loc[m, w]:.3f}" for w in p.columns) + " |")
        lines.append("")
    lines += ["## Hypotheses", "", "```", str(verdict), "```"]
    Path(cfg["output_dir"], "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
