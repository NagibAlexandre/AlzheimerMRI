import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "specificity", "fpr", "fnr"]

METRIC_LABELS = {
    "accuracy":    "Accuracy",
    "precision":   "Precision",
    "recall":      "Recall / Sensitivity",
    "f1":          "F1-Score",
    "specificity": "Specificity",
    "fpr":         "FPR",
    "fnr":         "FNR",
}

# Metrics shown in the boxplot (the most relevant for the paper)
BOXPLOT_KEYS = ["accuracy", "precision", "recall", "f1", "specificity"]



# Read data
def load_run_metrics(results_dir: Path) -> list[dict]:
    """Read all metricas_run_*.json files and return a list of final-metrics dicts."""
    run_files = sorted(results_dir.glob("metricas_run_*.json"))
    if not run_files:
        raise FileNotFoundError(
            f"No 'metricas_run_*.json' files found in:\n  {results_dir}\n"
            f"Run run_experiment.py first."
        )

    runs = []
    for fpath in run_files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if "final_metrics" not in data:
            print(f"WARNING: {fpath.name} does not contain 'final_metrics' — ignored.")
            continue
        runs.append({"run_id": data.get("run_id"), **data["final_metrics"]})

    print(f"Runs loaded: {len(runs)}  (of {len(run_files)} files)")
    return runs



# Statistics
def compute_stats(values: list[float], confidence: float = 0.95) -> dict:
    """Return mean, sample standard deviation and CI (t-distribution, n-1 df)."""
    n   = len(values)
    arr = np.array(values, dtype=float)

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n > 1:
        t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
        margin = t_crit * std / np.sqrt(n)
        ci_low  = float(mean - margin)
        ci_high = float(mean + margin)
    else:
        t_crit = margin = ci_low = ci_high = float("nan")

    return {
        "n":       n,
        "mean":    mean,
        "std":     std,
        "ci_low":  ci_low,
        "ci_high": ci_high,
        "t_crit":  t_crit,
        "margin":  margin,
    }


def build_stats(runs: list[dict]) -> dict[str, dict]:
    """Compute statistics for each metric."""
    result = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in runs if key in r and r[key] is not None]
        if vals:
            result[key] = compute_stats(vals)
    return result



# Table formatting
def format_table(stats_by_metric: dict[str, dict], as_percentage: bool = True) -> str:
    """Format a table ready to be copied into the paper."""
    n_runs = next(iter(stats_by_metric.values()))["n"] if stats_by_metric else 0

    col_metric = 22
    col_mean   = 9
    col_std    = 9
    col_ci     = 26
    total_w    = col_metric + col_mean + col_std + col_ci + 9  

    sep  = "-" * total_w
    header = (
        f"{'Metric':<{col_metric}} | "
        f"{'Mean':>{col_mean}} | "
        f"{'Std':>{col_std}} | "
        f"{'95% CI':>{col_ci}}"
    )

    lines = [sep, header, sep]

    for key in METRIC_KEYS:
        if key not in stats_by_metric:
            continue
        s     = stats_by_metric[key]
        label = METRIC_LABELS.get(key, key)

        if as_percentage:
            mean_s = f"{s['mean']*100:.2f}%"
            std_s  = f"{s['std']*100:.2f}%"
            ci_s   = f"[{s['ci_low']*100:.2f}%, {s['ci_high']*100:.2f}%]"
        else:
            mean_s = f"{s['mean']:.4f}"
            std_s  = f"{s['std']:.4f}"
            ci_s   = f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"

        lines.append(
            f"{label:<{col_metric}} | "
            f"{mean_s:>{col_mean}} | "
            f"{std_s:>{col_std}} | "
            f"{ci_s:>{col_ci}}"
        )

    lines.append(sep)
    lines.append(
        f"n = {n_runs} runs | "
        f"95% CI via t-distribution (df = n-1) | "
        f"Client metrics averaged across {3} clients/run"
    )
    return "\n".join(lines)



# Boxplot
def plot_boxplot(runs: list[dict], output_path: Path):
    """Generate boxplot of main metrics with individual points."""
    n_runs = len(runs)
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    data   = []
    labels = []
    for key in BOXPLOT_KEYS:
        vals = [r[key] * 100 for r in runs if key in r and r[key] is not None]
        if vals:
            data.append(vals)
            labels.append(METRIC_LABELS.get(key, key))

    if not data:
        print("WARNING: No data for the boxplot.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1.5},
                    capprops={"linewidth": 1.5})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Individual points (jitter)
    rng = np.random.default_rng(seed=None)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        x_jitter = rng.uniform(-0.12, 0.12, size=len(vals)) + i
        ax.scatter(x_jitter, vals, color=color, alpha=0.85,
                   zorder=5, s=40, edgecolors="white", linewidths=0.5)

    ax.set_ylabel("Value (%)", fontsize=12)
    ax.set_title(
        f"Federated Learning — Metrics Distribution\n"
        f"(n = {n_runs} independent runs, 3 clients each)",
        fontsize=13
    )
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.set_ylim(bottom=max(0, min(min(v) for v in data) - 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Boxplot saved to: {output_path}")
    plt.close()



# Main
def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of Federated Learning runs"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Directory with result JSONs (default: {RESULTS_DIR})"
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    print(f"\nStatistical analysis — Federated Learning")
    print(f"  Directory: {results_dir}")

    # --- Load data ---
    runs = load_run_metrics(results_dir)
    if not runs:
        print("No valid runs found. Aborting.")
        return

    # --- Statistics ---
    stats_by_metric = build_stats(runs)

    # --- Table ---
    table = format_table(stats_by_metric)
    print("\n" + table + "\n")

    out_txt = results_dir / "statistical_analysis.txt"
    out_txt.write_text(table, encoding="utf-8")
    print(f"✓ Table saved to: {out_txt}")

    # --- Raw stats JSON ---
    out_json = results_dir / "statistical_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {k: {sk: sv for sk, sv in v.items()} for k, v in stats_by_metric.items()},
            f, indent=4
        )
    print(f"✓ Stats JSON saved to: {out_json}")

    # --- Boxplot ---
    plot_boxplot(runs, results_dir / "boxplot_metrics.png")


if __name__ == "__main__":
    main()
