"""Graph construction per module presentation and engineered graph features.

* Bipartite student-resource graph: edge weight = log(1 + clicks before cut-off).
* Student-student projection: each student linked to the k most similar students in
  the same presentation (cosine similarity of IDF-weighted resource vectors),
  made undirected, with self-loops added later by the GNN layers.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .data_prep import ASSESSMENT, ENGAGEMENT, KEY
from .utils import LOG

NEIGHBOUR_FEATURES = ["clicks_total", "active_days", "days_since_last", "clicks_last_week",
                      "n_resources", "n_submitted", "submit_rate", "mean_score", "n_late"]
GRAPH_FEATURES = (["g_degree", "g_mean_sim", "g_clustering", "g_pagerank"] +
                  [f"nb_{c}" for c in NEIGHBOUR_FEATURES])


def _presentation_graph(students, clicks, k):
    """students: DataFrame of one presentation (index 0..n-1). clicks: rows of studentVle."""
    n = len(students)
    pos = pd.Series(np.arange(n), index=students["id_student"].to_numpy())
    clicks = clicks[clicks["id_student"].isin(pos.index)]
    e = clicks.groupby(["id_student", "id_site"], observed=True)["sum_click"].sum().reset_index()
    sites = pd.Series(np.arange(e["id_site"].nunique()), index=np.sort(e["id_site"].unique()))
    rows = pos.loc[e["id_student"]].to_numpy()
    cols = sites.loc[e["id_site"]].to_numpy()
    w = np.log1p(e["sum_click"].to_numpy(float))
    B = sparse.csr_matrix((w, (rows, cols)), shape=(n, len(sites)))

    # IDF weighting and cosine similarity
    df_ = np.asarray((B > 0).sum(axis=0)).ravel()
    idf = np.log((1 + n) / (1 + df_)) + 1.0
    X = B @ sparse.diags(idf)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = sparse.diags(1.0 / norms) @ X
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)

    src, dst, sim = [], [], []
    for i in range(n):
        row = S[i]
        if row.max() <= 0:
            continue
        kk = min(k, int((row > 0).sum()))
        nb = np.argpartition(-row, kk - 1)[:kk]
        src.extend([i] * kk)
        dst.extend(nb.tolist())
        sim.extend(row[nb].tolist())
    src, dst, sim = np.array(src, int), np.array(dst, int), np.array(sim, float)
    # undirected: keep max similarity for each unordered pair
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    und = pd.DataFrame({"a": a, "b": b, "s": sim}).groupby(["a", "b"], as_index=False)["s"].max()
    return B, und


def build_graphs(snapshot, student_vle, week, k):
    """Return (edges DataFrame with global row indices, graph-feature DataFrame)."""
    cutoff = 7 * week
    sv = student_vle[student_vle["date"] < cutoff]
    sv_groups = {key: g for key, g in sv.groupby(["code_module", "code_presentation"], observed=True)}
    all_edges, feats = [], []
    for (mod, pres), idx in snapshot.groupby(["code_module", "code_presentation"]).groups.items():
        students = snapshot.loc[idx].reset_index()
        clicks = sv_groups.get((mod, pres), sv.iloc[0:0])
        B, und = _presentation_graph(students, clicks, k)
        n = len(students)
        gidx = students["index"].to_numpy()
        all_edges.append(pd.DataFrame({"u": gidx[und["a"]], "v": gidx[und["b"]], "s": und["s"]}))

        # projection features
        deg = np.bincount(und["a"], minlength=n) + np.bincount(und["b"], minlength=n)
        ssum = np.bincount(und["a"], weights=und["s"], minlength=n) + np.bincount(und["b"], weights=und["s"], minlength=n)
        A = sparse.csr_matrix((np.ones(len(und)), (und["a"], und["b"])), shape=(n, n))
        A = ((A + A.T) > 0).astype(float)
        tri = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2
        possible = deg * (deg - 1) / 2
        clustering = np.divide(tri, possible, out=np.zeros(n), where=possible > 0)

        # PageRank in the weighted bipartite graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        Bc = B.tocoo()
        G.add_weighted_edges_from(zip(Bc.row.tolist(), (Bc.col + n).tolist(), Bc.data.tolist()))
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}
        pagerank = np.array([pr.get(i, 0.0) for i in range(n)]) * n

        # neighbour feature means (one-hop aggregation)
        vals = np.log1p(students[NEIGHBOUR_FEATURES].clip(lower=0).to_numpy(float))
        degs = np.asarray(A.sum(axis=1)).ravel()
        nb_mean = np.divide(A @ vals, degs[:, None], out=np.zeros_like(vals), where=degs[:, None] > 0)

        f = pd.DataFrame(nb_mean, columns=[f"nb_{c}" for c in NEIGHBOUR_FEATURES])
        f["g_degree"] = deg
        f["g_mean_sim"] = np.divide(ssum, deg, out=np.zeros(n), where=deg > 0)
        f["g_clustering"] = clustering
        f["g_pagerank"] = pagerank
        f.index = gidx
        feats.append(f)
    edges = pd.concat(all_edges, ignore_index=True)
    gf = pd.concat(feats).sort_index()[GRAPH_FEATURES]
    LOG.info("Week %2d graphs: %d students, %d undirected edges (mean degree %.1f)",
             week, len(snapshot), len(edges), 2 * len(edges) / len(snapshot))
    return edges, gf
