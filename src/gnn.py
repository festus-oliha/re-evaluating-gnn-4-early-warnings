"""Graph neural networks (GCN, GraphSAGE, GAT) trained with the same Optuna budget.

Split A (transductive): one graph containing all presentations; labels of validation
and test nodes are hidden during training.
Split B (inductive): the model is trained on the graphs of the training presentations
and applied to the separate graphs of the validation and test presentations.
"""
import copy

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .utils import set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(max(1, torch.get_num_threads()))


class GNN(nn.Module):
    def __init__(self, kind, in_dim, hidden, dropout, heads=4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        if kind == "GCN":
            self.c1, self.c2, d1 = (GCNConv(in_dim, hidden, normalize=False),
                                    GCNConv(hidden, hidden, normalize=False), hidden)
        elif kind == "GraphSAGE":
            self.c1, self.c2, d1 = SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden), hidden
        else:
            # hidden units are shared across heads so the layer width matches GCN and GraphSAGE
            self.c1 = GATConv(in_dim, hidden // heads, heads=heads, concat=True)
            self.c2 = GATConv(hidden, hidden, heads=1)
            d1 = hidden
        self.b1, self.b2 = nn.BatchNorm1d(d1), nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, g):
        if self.kind == "GCN":   # normalisation with self-loops is precomputed once per graph
            ei, ew = g.gcn_index, g.gcn_weight
            h = self.c1(g.x, ei, ew)
            h = nn.functional.dropout(torch.relu(self.b1(h)), self.dropout, self.training)
            h = self.c2(h, ei, ew)
        else:
            h = self.c1(g.x, g.edge_index)
            h = nn.functional.dropout(torch.relu(self.b1(h)), self.dropout, self.training)
            h = self.c2(h, g.edge_index)
        h = nn.functional.dropout(torch.relu(self.b2(h)), self.dropout, self.training)
        return self.out(h).squeeze(-1)


def _subgraph_data(X, y, edges, nodes):
    """Graph restricted to `nodes` (edges never cross presentations, so no edges are lost)."""
    pos = -np.ones(len(X), dtype=int)
    pos[nodes] = np.arange(len(nodes))
    keep = (pos[edges["u"].to_numpy()] >= 0) & (pos[edges["v"].to_numpy()] >= 0)
    u, v = pos[edges["u"].to_numpy()[keep]], pos[edges["v"].to_numpy()[keep]]
    ei = torch.tensor(np.vstack([np.concatenate([u, v]), np.concatenate([v, u])]), dtype=torch.long)
    g = Data(x=torch.tensor(X[nodes], dtype=torch.float32), edge_index=ei,
             y=torch.tensor(y[nodes], dtype=torch.float32))
    g.gcn_index, g.gcn_weight = gcn_norm(ei, None, len(nodes), add_self_loops=True)
    return g


def prepare(X, y, edges, split, design):
    """Standardise features (fitted on training nodes) and build graph objects."""
    Xv = X.to_numpy(np.float64)
    scaler = StandardScaler().fit(Xv[split["train"]])
    Xv = np.nan_to_num(scaler.transform(Xv)).astype(np.float32)
    if design == "A":
        allnodes = np.arange(len(Xv))
        g = _subgraph_data(Xv, y, edges, allnodes)
        return {"design": "A", "train": (g, split["train"]), "validation": (g, split["validation"]),
                "test": (g, split["test"])}
    out = {"design": "B"}
    for part in ("train", "validation", "test"):
        nodes = split[part]
        out[part] = (_subgraph_data(Xv, y, edges, nodes), np.arange(len(nodes)))
    return out


def _predict(model, graph, idx):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(graph))[idx].numpy()


def train_gnn(kind, data, params, seed, cfg):
    set_seed(seed)
    g_tr, i_tr = data["train"]
    g_va, i_va = data["validation"]
    ytr = g_tr.y[i_tr]
    pos_weight = torch.tensor((ytr == 0).sum().item() / max((ytr == 1).sum().item(), 1))
    model = GNN(kind, g_tr.x.shape[1], params["hidden"], params["dropout"], cfg["gnn"]["gat_heads"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    i_tr_t = torch.tensor(i_tr)
    yva = g_va.y[i_va].numpy()
    best, best_state, wait = -1.0, None, 0
    every = cfg["gnn"].get("eval_every", 2)
    for epoch in range(cfg["gnn"]["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(g_tr)[i_tr_t], ytr)
        loss.backward()
        opt.step()
        if epoch % every:
            continue
        score = average_precision_score(yva, _predict(model, g_va, i_va))
        if score > best + 1e-4:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += every
            if wait >= cfg["gnn"]["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best


def run_gnn(kind, data, cfg):
    def objective(trial):
        params = {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
                  "dropout": trial.suggest_float("dropout", 0.0, 0.6),
                  "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)}
        return train_gnn(kind, data, params, 0, cfg)[1]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["tuning"]["n_trials"])
    out = {"params": study.best_params, "val": [], "test": [], "models": []}
    for seed in cfg["tuning"]["seeds"]:
        model, _ = train_gnn(kind, data, study.best_params, seed, cfg)
        out["val"].append(_predict(model, *data["validation"]))
        out["test"].append(_predict(model, *data["test"]))
        out["models"].append(model)
    return out
