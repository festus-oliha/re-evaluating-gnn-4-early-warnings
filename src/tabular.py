"""Tabular models (logistic regression, random forest, XGBoost) with an equal Optuna budget."""
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make(name, params, seed, pos_weight):
    if name == "LR":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=3000)
    if name == "RF":
        mf = params["max_features"]
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"],
                                      max_features=float(mf) if mf not in ("sqrt", "log2") else mf,
                                      class_weight="balanced", n_jobs=-1, random_state=seed)
    return XGBClassifier(**params, scale_pos_weight=pos_weight, tree_method="hist", n_jobs=-1,
                         eval_metric="aucpr", random_state=seed, verbosity=0)


def _suggest(name, trial):
    if name == "LR":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "RF":
        depth = trial.suggest_categorical("max_depth", [4, 6, 8, 12, 16, 20, None])
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": depth,
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", "0.3", "0.5", "0.8"])}
    return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def run_tabular(name, X, y, split, cfg):
    """Tune on validation PR-AUC, then refit with each seed. Returns probabilities per seed."""
    tr, va, te = split["train"], split["validation"], split["test"]
    Xv = X.to_numpy(np.float32)
    if name == "LR":
        scaler = StandardScaler().fit(Xv[tr])
        Xv = scaler.transform(Xv)
    pos_weight = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))

    def objective(trial):
        model = _make(name, _suggest(name, trial), 0, pos_weight)
        model.fit(Xv[tr], y[tr])
        return average_precision_score(y[va], model.predict_proba(Xv[va])[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["tuning"]["n_trials"])
    params = _suggest(name, optuna.trial.FixedTrial(study.best_params))
    out = {"params": study.best_params, "val": [], "test": [], "models": []}
    for seed in cfg["tuning"]["seeds"]:
        model = _make(name, params, seed, pos_weight)
        model.fit(Xv[tr], y[tr])
        out["val"].append(model.predict_proba(Xv[va])[:, 1])
        out["test"].append(model.predict_proba(Xv[te])[:, 1])
        out["models"].append(model)
    return out
