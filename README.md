# Re-evaluating Graph Neural Networks for Early At-Risk Student Prediction

Code for my Student's [Bolaji, Co-supervised by Dr Nwelih] MSc Computer Science dissertation:
*Re-evaluating Graph Neural Networks for Early At-Risk Student Prediction: Temporal
Validation, Class Imbalance and Algorithmic Fairness.*

The project compares graph neural networks (GCN, GraphSAGE, GAT) with equally tuned
tabular models (logistic regression, random forest, XGBoost, and XGBoost with engineered
graph features) on the Open University Learning Analytics Dataset (OULAD). Models are
evaluated at weeks 2, 4, 6, 8 and 12 under a random split and a time-respecting split
(train on 2013, test on 2014), with imbalance-aware metrics and group fairness analysis.

## Research questions

1. Under a time-respecting split, do GNNs outperform equally tuned gradient boosting, and how
   does the gap change from week 2 to week 12?
2. How much of any GNN advantage disappears when tabular models are given engineered graph features?
3. How much does performance drop when moving from a random split to a 2013 → 2014 split?
4. Do GNNs and gradient boosting differ in fairness across demographic groups, and can group-specific
   thresholds reduce the gaps?

## Repository structure

```
config.yaml            all settings (snapshots, splits, tuning budget, seeds)
requirements.txt
data/                  put the seven OULAD CSV files here (not included)
src/
  data_prep.py         loading OULAD and building leakage-free snapshots
  graphs.py            student-resource graphs, kNN projection, graph features
  features.py          design matrices and Split A / Split B
  tabular.py           LR, RF, XGBoost with Optuna tuning
  gnn.py               GCN, GraphSAGE, GAT (PyTorch Geometric)
  experiment.py        runs all models for one week and caches predictions
  metrics.py           PR-AUC, recall at budget, Brier, bootstrap
  analysis.py          tables, hypothesis tests, fairness and mitigation
  explain.py           SHAP and GNNExplainer
  figures.py           figures
  run_all.py           entry point
outputs/               tables, figures, hypotheses.json, results_summary.md, run_log.txt
```

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

On Windows or a computer without an NVIDIA GPU, a smaller CPU-only PyTorch can be installed first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Data

Download OULAD from https://analyse.kmi.open.ac.uk/open-dataset and place these files in `data/`:

`courses.csv`, `assessments.csv`, `vle.csv`, `studentInfo.csv`, `studentRegistration.csv`,
`studentAssessment.csv`, `studentVle.csv`.

OULAD is released under a Creative Commons Attribution 4.0 licence (Kuzilek, Hlosta and Zdrahal, 2017).

## Running

```bash
python -m src.run_all --quick           # smoke test, a few minutes
python -m src.run_all                   # full experiment
python -m src.run_all --weeks 4 8       # selected weeks only
python -m src.run_all --analysis-only   # rebuild tables and figures from cached results
```

Results for each week are cached in `outputs/cache/`, so an interrupted run can be restarted
and completed weeks are skipped (use `--force` to recompute). The full run takes about two
hours on a two-core laptop without a GPU; it is faster on machines with more cores.

## Outputs

| File | Content |
| --- | --- |
| `tables/t01_data_summary.csv` | Snapshot sizes, at-risk rates, split sizes, graph sizes |
| `tables/t03_performance_summary.csv` | Test metrics, mean and sd over 5 seeds, both splits |
| `tables/t04_performance_ensemble.csv` | Seed-averaged predictions with bootstrap CIs |
| `tables/t05_rq1_gnn_vs_xgb.csv` | Best GNN vs XGBoost (paired bootstrap) |
| `tables/t06_h1.csv`, `t07_h2_advantage.csv`, `t11_h3.csv` | Hypothesis tests |
| `tables/t09_split_drop.csv` | PR-AUC drop from Split A to Split B |
| `tables/t10_fairness.csv`, `t12_mitigation.csv` | Fairness metrics and mitigation |
| `tables/t13_without_ccc.csv`, `t14_ablation_demographics.csv` | Robustness checks |
| `tables/t15_shap.csv`, `t16_gnnexplainer.csv` | Explanations |
| `hypotheses.json` | Verdicts for H1–H3 |
| `figures/*.png` | Figures used in the dissertation |

## Licence

Code: MIT. Data: OULAD, CC BY 4.0.
