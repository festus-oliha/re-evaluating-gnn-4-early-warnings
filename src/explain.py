"""SHAP explanations for the gradient-boosting model."""
import numpy as np
import pandas as pd


def shap_explain(model, X, test_idx):
    import shap
    Xt = X.iloc[test_idx]
    sv = shap.TreeExplainer(model).shap_values(Xt.to_numpy(np.float32))
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return imp
