# Results summary

## Test PR-AUC (mean ± sd over seeds)

### Split B
| Model | Week 4 | Week 8 |
|---|---|---|
| LR | 0.638 ± 0.000 | 0.743 ± 0.000 |
| RF | 0.660 ± 0.006 | 0.714 ± 0.001 |
| XGB | 0.634 ± 0.004 | 0.684 ± 0.001 |
| XGB+graph | 0.635 ± 0.007 | 0.709 ± 0.003 |
| GCN | 0.625 ± 0.007 | 0.650 ± 0.001 |
| GraphSAGE | 0.678 ± 0.001 | 0.717 ± 0.003 |
| GAT | 0.625 ± 0.005 | 0.654 ± 0.005 |

### Split A
| Model | Week 4 | Week 8 |
|---|---|---|
| LR | 0.760 ± 0.000 | 0.806 ± 0.000 |
| RF | 0.762 ± 0.000 | 0.808 ± 0.002 |
| XGB | 0.750 ± 0.005 | 0.793 ± 0.005 |
| XGB+graph | 0.744 ± 0.001 | 0.797 ± 0.002 |
| GCN | 0.675 ± 0.008 | 0.704 ± 0.003 |
| GraphSAGE | 0.725 ± 0.000 | 0.760 ± 0.002 |
| GAT | 0.681 ± 0.002 | 0.710 ± 0.004 |

## Hypotheses

```
{'H1': {'weeks_matching': 1, 'n_weeks': 2, 'supported': False}, 'H2': {'weeks_shrinking': 0, 'n_weeks': 2, 'supported': False}, 'H3': {'comparisons': 8, 'gnn_significantly_fairer': 1, 'supported': True}}
```