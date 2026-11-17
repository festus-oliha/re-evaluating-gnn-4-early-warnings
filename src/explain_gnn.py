"""Post-hoc explanations of the best GNN at each week (Split B).

The best GNN is retrained with seed 0 and its tuned hyperparameters, then:
  * graph reliance: predictions with and without the student-student edges are compared
    for every test student (how much the model depends on neighbours);
  * permutation importance: drop in test PR-AUC when one feature is shuffled across test
    students (all weeks), comparable with SHAP for gradient boosting;
  * GNNExplainer (selected weeks): feature-mask importance of each explained student's own
    features, averaged over a sample of test students with predicted risk >= 0.5.
Results are stored in the week's cache file.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .data_prep import build_snapshot
from .features import design_matrix, make_splits
from .gnn import _predict, prepare, train_gnn
from .graphs import build_graphs
from .metrics import best_f1_threshold
from .utils import LOG


def _without_edges(g):
    h = g.clone()
    h.edge_index = torch.empty((2, 0), dtype=torch.long)
    n = h.x.shape[0]
    loops = torch.arange(n)
    h.gcn_index, h.gcn_weight = torch.stack([loops, loops]), torch.ones(n)
    return h


def gnnexplainer_features(model, graph, feature_names, cfg):
    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import k_hop_subgraph

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, edge_index):
            g = Data(x=x, edge_index=edge_index)
            g.gcn_index, g.gcn_weight = gcn_norm(edge_index, None, x.shape[0], add_self_loops=True)
            return self.m(g)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(graph)).numpy()
    cand = np.where(prob >= 0.5)[0]
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.choice(cand, size=min(cfg["explain"]["gnnexplainer_students"], len(cand)), replace=False)
    explainer = Explainer(model=Wrapper(model), algorithm=GNNExplainer(epochs=cfg["explain"]["gnnexplainer_epochs"]),
                          explanation_type="model", node_mask_type="attributes", edge_mask_type=None,
                          model_config=dict(mode="binary_classification", task_level="node", return_type="raw"))
    own = np.zeros(len(feature_names))
    for node in sample:
        subset, sub_ei, mapping, _ = k_hop_subgraph(int(node), 2, graph.edge_index, relabel_nodes=True)
        e = explainer(graph.x[subset], sub_ei, index=int(mapping[0]))
        own += e.node_mask[int(mapping[0])].detach().numpy()
    return pd.Series(own / len(sample), index=feature_names).sort_values(ascending=False), int(len(sample))


def run_gnn_explanations(tables, cfg, force=False):
    cache = Path(cfg["output_dir"], "cache")
    for week in cfg["snapshots"]:
        f = cache / f"week_{week}.pkl"
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if "graph_reliance" in d["explain"] and not force:
            continue
        snap = build_snapshot(tables, week)
        edges, _ = build_graphs(snap, tables["studentVle"], week, cfg["graph"]["k_neighbours"])
        X = design_matrix(snap)
        y = snap["at_risk"].to_numpy()
        data = prepare(X, y, edges, make_splits(snap, cfg)["B"], "B")
        kind = d["best_gnn"]
        model, _ = train_gnn(kind, data, d["params"][kind], 0, cfg)
        gv, iv = data["validation"]
        gt, it = data["test"]
        pv = _predict(model, gv, iv)
        thr = best_f1_threshold(gv.y.numpy(), pv)
        p_full = _predict(model, gt, it)
        p_none = _predict(model, _without_edges(gt), it)
        yt = gt.y.numpy()
        rel = {"model": kind, "pr_auc_with_edges": float(average_precision_score(yt, p_full)),
               "pr_auc_without_edges": float(average_precision_score(yt, p_none)),
               "mean_abs_change": float(np.abs(p_full - p_none).mean()),
               "flag_changed_share": float(((p_full >= thr) != (p_none >= thr)).mean()),
               "n_test": int(len(yt))}
        d["explain"]["graph_reliance"] = rel
        rng = np.random.default_rng(cfg["seed"])
        base = rel["pr_auc_with_edges"]
        drops = {}
        for j, name in enumerate(X.columns):
            vals = []
            for _ in range(3):
                g2 = gt.clone()
                g2.x = gt.x.clone()
                g2.x[:, j] = gt.x[rng.permutation(gt.x.shape[0]), j]
                vals.append(base - average_precision_score(yt, _predict(model, g2, it)))
            drops[name] = float(np.mean(vals))
        d["explain"]["gnn_permutation"] = pd.Series(drops).sort_values(ascending=False)
        LOG.info("Week %d %s graph reliance: PR-AUC %.3f with edges, %.3f without; mean |change| %.3f; flags changed %.1f%%",
                 week, kind, rel["pr_auc_with_edges"], rel["pr_auc_without_edges"], rel["mean_abs_change"],
                 100 * rel["flag_changed_share"])
        if week in cfg["explain"]["gnnexplainer_weeks"]:
            imp, n = gnnexplainer_features(model, gt, list(X.columns), cfg)
            d["explain"]["gnnexplainer"] = {"feature_importance": imp, "n_students": n}
            LOG.info("Week %d GNNExplainer top features: %s", week, ", ".join(imp.index[:5]))
        with open(f, "wb") as fh:
            pickle.dump(d, fh)
