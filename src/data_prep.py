"""Load OULAD and build leakage-free feature snapshots.

For a snapshot at week w the cut-off day is c = 7w. Only clicks and assessment
submissions dated before c are used, and students who unregistered before c are
removed (they have already withdrawn).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import LOG

KEY = ["code_module", "code_presentation", "id_student"]
MAIN_TYPES = ["oucontent", "resource", "forumng", "quiz", "homepage", "subpage", "url"]
DEMOGRAPHIC_CATS = ["gender", "age_band", "highest_education", "imd_band", "disability", "region"]
DEMOGRAPHIC_NUMS = ["num_of_prev_attempts", "studied_credits"]
ENGAGEMENT = (["clicks_total"] + [f"clicks_{t}" for t in MAIN_TYPES] +
              ["clicks_other", "active_days", "days_since_last", "clicks_last_week",
               "click_trend", "pre_start_clicks", "n_resources"])
ASSESSMENT = ["n_due", "n_submitted", "submit_rate", "mean_score", "n_late"]
COUNT_FEATURES = ([c for c in ENGAGEMENT if c.startswith("clicks_") and c != "click_trend"] +
                  ["pre_start_clicks", "active_days", "n_resources"])


def load_oulad(data_dir):
    d = Path(data_dir)
    needed = ["courses", "assessments", "vle", "studentInfo", "studentRegistration",
              "studentAssessment", "studentVle"]
    missing = [n for n in needed if not (d / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing OULAD files in {d.resolve()}: {missing}")
    LOG.info("Loading OULAD from %s", d)
    tables = {n: pd.read_csv(d / f"{n}.csv") for n in needed if n != "studentVle"}
    tables["studentVle"] = pd.read_csv(
        d / "studentVle.csv",
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"})
    for name, t in tables.items():
        LOG.info("  %-20s %10s rows", name, f"{len(t):,}")
    return tables


def _slope(weekly):
    """Least-squares slope of each row of a (students x weeks) matrix."""
    n = weekly.shape[1]
    if n < 2:
        return np.zeros(weekly.shape[0])
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    return (weekly - weekly.mean(axis=1, keepdims=True)) @ t / (t ** 2).sum()


def build_snapshot(tables, week):
    """Return one row per registration with features, label and group columns."""
    cutoff = 7 * week
    info = tables["studentInfo"].merge(tables["studentRegistration"], on=KEY, how="left")
    withdrawn_early = info["date_unregistration"].notna() & (info["date_unregistration"] < cutoff)
    df = info.loc[~withdrawn_early].copy()
    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    # ---- engagement -------------------------------------------------------
    sv = tables["studentVle"]
    sv = sv[sv["date"] < cutoff]
    vle = tables["vle"][["id_site", "activity_type"]]
    sv = sv.merge(vle, on="id_site", how="left")
    sv["atype"] = np.where(sv["activity_type"].isin(MAIN_TYPES), sv["activity_type"], "other")
    pre = sv[sv["date"] < 0].groupby(KEY, observed=True)["sum_click"].sum().rename("pre_start_clicks")
    cur = sv[sv["date"] >= 0]
    by_type = cur.pivot_table(index=KEY, columns="atype", values="sum_click", aggfunc="sum",
                              fill_value=0, observed=True)
    by_type.columns = [f"clicks_{c}" for c in by_type.columns]
    agg = cur.groupby(KEY, observed=True).agg(clicks_total=("sum_click", "sum"),
                                              active_days=("date", "nunique"))
    last = sv.groupby(KEY, observed=True)["date"].max().rename("last_date")
    nres = sv.groupby(KEY, observed=True)["id_site"].nunique().rename("n_resources")
    lastw = cur[cur["date"] >= cutoff - 7].groupby(KEY, observed=True)["sum_click"].sum().rename("clicks_last_week")
    cur = cur.assign(wk=(cur["date"] // 7).astype(int))
    weekly = cur.pivot_table(index=KEY, columns="wk", values="sum_click", aggfunc="sum",
                             fill_value=0, observed=True)
    weekly = weekly.reindex(columns=range(week), fill_value=0)
    trend = pd.Series(_slope(np.log1p(weekly.to_numpy(float))), index=weekly.index, name="click_trend")
    eng = pd.concat([agg, by_type, pre, last, nres, lastw, trend], axis=1).reset_index()
    eng["code_module"] = eng["code_module"].astype(str)
    eng["code_presentation"] = eng["code_presentation"].astype(str)
    df = df.merge(eng, on=KEY, how="left")
    for c in ENGAGEMENT:
        if c not in df:
            df[c] = 0.0
    df["days_since_last"] = np.where(df["last_date"].isna(), cutoff + 1, cutoff - df["last_date"])
    fill = [c for c in ENGAGEMENT if c != "days_since_last"]
    df[fill] = df[fill].fillna(0)

    # ---- assessment -------------------------------------------------------
    ass = tables["assessments"]
    ass = ass[(ass["assessment_type"] != "Exam") & ass["date"].notna() & (ass["date"] < cutoff)]
    n_due = ass.groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()
    sa = tables["studentAssessment"].merge(ass[["id_assessment", "code_module", "code_presentation", "date"]],
                                          on="id_assessment", how="inner")
    sa = sa[sa["date_submitted"] < cutoff]
    sa = sa.assign(late=(sa["date_submitted"] > sa["date"]).astype(int))
    sub = sa.groupby(KEY).agg(n_submitted=("id_assessment", "nunique"), mean_score=("score", "mean"),
                              n_late=("late", "sum")).reset_index()
    df = df.merge(n_due, on=["code_module", "code_presentation"], how="left").merge(sub, on=KEY, how="left")
    df["n_due"] = df["n_due"].fillna(0)
    df[["n_submitted", "n_late"]] = df[["n_submitted", "n_late"]].fillna(0)
    df["mean_score"] = df["mean_score"].fillna(0)
    df["submit_rate"] = np.where(df["n_due"] > 0, df["n_submitted"] / df["n_due"].clip(lower=1), 0.0)

    # ---- demographics and protected groups --------------------------------
    df["imd_band"] = df["imd_band"].fillna("Missing").replace({"10-20": "10-20%"})
    df["grp_gender"] = np.where(df["gender"] == "F", "interest", "reference")
    df["grp_age"] = np.where(df["age_band"] == "0-35", "reference", "interest")
    df["grp_disability"] = np.where(df["disability"] == "Y", "interest", "reference")
    deprived = df["imd_band"].isin(["0-10%", "10-20%", "20-30%"])
    df["grp_imd"] = pd.Series(np.where(deprived, "interest", "reference"), index=df.index).where(df["imd_band"] != "Missing")
    df["week"] = week
    keep = (KEY + ["week", "at_risk", "final_result"] + ENGAGEMENT + ASSESSMENT + DEMOGRAPHIC_CATS +
            DEMOGRAPHIC_NUMS + ["grp_gender", "grp_age", "grp_disability", "grp_imd"])
    df = df[keep].reset_index(drop=True)
    LOG.info("Week %2d: %6d registrations, %.1f%% at risk (removed %d early withdrawals)",
             week, len(df), 100 * df["at_risk"].mean(), int(withdrawn_early.sum()))
    return df


def leakage_check(snapshots):
    """Cumulative features must never decrease between successive snapshots."""
    cumulative = ["clicks_total", "active_days", "pre_start_clicks", "n_resources", "n_due", "n_submitted"]
    rows = []
    weeks = sorted(snapshots)
    for a, b in zip(weeks[:-1], weeks[1:]):
        m = snapshots[a].merge(snapshots[b], on=KEY, suffixes=("_a", "_b"))
        for c in cumulative:
            bad = int((m[f"{c}_b"] < m[f"{c}_a"] - 1e-9).sum())
            rows.append({"from_week": a, "to_week": b, "feature": c, "violations": bad, "n": len(m)})
    out = pd.DataFrame(rows)
    LOG.info("Leakage check: %d violations across %d comparisons", int(out["violations"].sum()), len(out))
    return out
