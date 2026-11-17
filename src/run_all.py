"""Entry point.

    python -m src.run_all                 # full run (experiments + analysis)
    python -m src.run_all --quick         # fast smoke test (2 trials, 2 seeds, weeks 4 and 8)
    python -m src.run_all --weeks 4 8     # selected weeks only
    python -m src.run_all --analysis-only # rebuild tables and figures from cached results
"""
import argparse
from pathlib import Path

from .utils import LOG, Timer, load_config, set_seed, setup_logging


def main():
    ap = argparse.ArgumentParser(description="GNN vs gradient boosting benchmark on OULAD")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--weeks", nargs="+", type=int)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute weeks that are already cached")
    ap.add_argument("--analysis-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, data_dir=args.data_dir, output_dir=args.output_dir)
    if args.quick:
        cfg["output_dir"] = args.output_dir or "outputs_quick"
        cfg["snapshots"] = [4, 8]
        cfg["tuning"].update(n_trials=2, seeds=[0, 1])
        cfg["gnn"].update(max_epochs=20, patience=6)
        cfg["evaluation"]["n_bootstrap"] = 100
        cfg["explain"].update(gnnexplainer_students=5, gnnexplainer_epochs=20)
    if args.weeks:
        cfg["snapshots"] = args.weeks
    setup_logging(cfg["output_dir"])
    set_seed(cfg["seed"])

    if not args.analysis_only:
        from .data_prep import leakage_check, load_oulad, build_snapshot
        from .experiment import run_week
        tables = load_oulad(cfg["data_dir"])
        cache = Path(cfg["output_dir"]) / "cache"
        snaps = {w: build_snapshot(tables, w) for w in cfg["snapshots"]}
        if len(snaps) > 1:
            Path(cfg["output_dir"], "tables").mkdir(parents=True, exist_ok=True)
            leakage_check(snaps).to_csv(Path(cfg["output_dir"], "tables", "leakage_check.csv"), index=False)
        for week in cfg["snapshots"]:
            if (cache / f"week_{week}.pkl").exists() and not args.force:
                LOG.info("Week %d already cached; skipping (use --force to recompute)", week)
                continue
            with Timer(f"Week {week}"):
                run_week(tables, week, cfg)
        from .explain_gnn import run_gnn_explanations
        with Timer("GNN explanations"):
            run_gnn_explanations(tables, cfg, force=args.force)

    from .analysis import run_analysis
    with Timer("Analysis"):
        run_analysis(cfg)


if __name__ == "__main__":
    main()
