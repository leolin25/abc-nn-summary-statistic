from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# change run date between experiment folders
RUN_DATE = "27-11-26"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / RUN_DATE

true_params = {
    "mu": (0.1, r"$\mu$"),
    "sigma": (0.2, r"$\sigma$"),
    "tau": (0.5, r"$\tau$"),
}

# total_sim xlims for each row
rows = [1e5, 1e6, 1e7]

# marker styling
METHOD_STYLES = {
    "model": {"color": "black", "marker": "s", "label": "model-based"},
    "canonical": {"color": "red", "marker": "o", "label": "canonical"},
    "nn": {"color": "blue", "marker": "^", "label": "PEN"},
}

def load_abc_results(data_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Import data from csv files
    """
    if not data_dir.exists():
        available = (
            sorted(p.name for p in data_dir.parent.iterdir() if p.is_dir())
            if data_dir.parent.exists()
            else []
        )
        raise FileNotFoundError(
            f"Folder not found: {data_dir}\nAvailable folders: {available}"
        )

    results = {}
    metrics_list = []
    particles_list = []

    for metrics_path in sorted(data_dir.glob("*_round_metrics.csv")):
        method = metrics_path.name.removesuffix("_round_metrics.csv")
        particles_path = data_dir / f"{method}_particles_weights.csv"

        if not particles_path.exists():
            raise FileNotFoundError(f"Missing matching particles file: {particles_path}")

        df_metrics = pd.read_csv(metrics_path)
        df_particles = pd.read_csv(particles_path)

        results[method] = {"metrics": df_metrics, "particles": df_particles}
        metrics_list.append(df_metrics.assign(method=method))
        particles_list.append(df_particles.assign(method=method))

    if not results:
        raise FileNotFoundError(f"No '*_round_metrics.csv' files found in {data_dir}")

    return (
        results,
        pd.concat(metrics_list, ignore_index=True),
        pd.concat(particles_list, ignore_index=True),
    )


def weighted_quantiles(
    values: np.ndarray, quantiles: list[float], weights: np.ndarray
) -> np.ndarray:
    """Compute exact weighted quantiles for importance-weighted SMC particles."""
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    w = w / np.sum(w)
    cdf = np.cumsum(w) - 0.5 * w
    return np.interp(quantiles, cdf, v)


def compute_ci_trajectories(results: dict, param_names: list[str]) -> dict:
    """
    Compute 5%, 50% and 95% quantiles of the weighted posterior
    """
    ci_data = {}
    for method, data in results.items():
        df_m = data["metrics"].sort_values("round")
        df_p = data["particles"]

        rounds = df_m["round"].to_numpy()
        cum_sims = df_m["cum_total_sims"].to_numpy()

        ci_data[method] = {"cum_sims": cum_sims}
        for param in param_names:
            q05, q50, q95 = [], [], []
            for r in rounds:
                sub = df_p[df_p["round"] == r]
                vals = sub[param].to_numpy()
                wts = sub["weight"].to_numpy()
                low, med, high = weighted_quantiles(vals, [0.05, 0.50, 0.95], wts)
                q05.append(low)
                q50.append(med)
                q95.append(high)

            ci_data[method][param] = {
                "q05": np.array(q05),
                "q50": np.array(q50),
                "q95": np.array(q95),
            }
    return ci_data


def _sci_formatter(x: float, _) -> str:
    """
    Format x labels to sci format (e.g 1e+07)
    """
    if x == 0:
        return "0e+00"
    return f"{x:.1e}".replace(".0e", "e")


def plot(
    ci_data: dict,
    true_params: dict,
    budget_rows: list[float],
    ylims: dict | None = None,
    save_path: Path | None = None,
):
    """Plot 90% credible intervals and medians across multiple computational budgets."""
    param_keys = list(true_params.keys())
    n_rows = len(budget_rows)
    n_cols = len(param_keys)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.7 * n_rows),
        sharey="col",
    )
    axes = np.atleast_2d(axes)

    ordered_methods = [m for m in ["model", "canonical", "nn"] if m in ci_data] + [
        m for m in ci_data if m not in {"model", "canonical", "nn"}
    ]

    for c, param in enumerate(param_keys):
        true_val, latex_label = true_params[param]

        # Determine consistent y-limits per column (clipping wide Round 1 intervals)
        if ylims and ylims.get(param) is not None:
            col_ymin, col_ymax = ylims[param]
        else:
            all_lows = [
                ci_data[m][param]["q05"][1:] if len(ci_data[m][param]["q05"]) > 1 else ci_data[m][param]["q05"]
                for m in ordered_methods
            ]
            all_highs = [
                ci_data[m][param]["q95"][1:] if len(ci_data[m][param]["q95"]) > 1 else ci_data[m][param]["q95"]
                for m in ordered_methods
            ]
            ymin = min(np.min(np.concatenate(all_lows)), true_val)
            ymax = max(np.max(np.concatenate(all_highs)), true_val)
            pad = 0.12 * (ymax - ymin)
            col_ymin, col_ymax = ymin - pad, ymax + pad

        for r, max_budget in enumerate(budget_rows):
            ax = axes[r, c]

            # true param line
            ax.axhline(true_val, color="#32CD32", linestyle="-", linewidth=1.3, zorder=1)

            for method in ordered_methods:
                style = METHOD_STYLES.get(
                    method, {"color": "gray", "marker": "d", "label": method}
                )
                sims = ci_data[method]["cum_sims"]
                q05 = ci_data[method][param]["q05"]
                q50 = ci_data[method][param]["q50"]
                q95 = ci_data[method][param]["q95"]

                ax.plot(
                    sims,
                    q95,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.8,
                    linewidth=1.0,
                    label=style["label"] if (r == 0 and c == 0) else "_nolegend_",
                    zorder=3,
                )
                ax.plot(
                    sims,
                    q05,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.8,
                    linewidth=1.0,
                    label="_nolegend_",
                    zorder=3,
                )
                ax.plot(
                    sims,
                    q50,
                    color=style["color"],
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.85,
                    label="_nolegend_",
                    zorder=2,
                )

            # set xlim and ylim and add sci formatting
            ax.set_xlim(0, max_budget)
            ax.set_ylim(col_ymin, col_ymax)
            ax.xaxis.set_major_formatter(FuncFormatter(_sci_formatter))
            ax.tick_params(axis="both", labelsize=8.5, direction="in", top=True, right=True)

            # parameter symbol in top-right corner
            ax.text(
                0.94,
                0.90,
                latex_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
            )

            if c == 0:
                ax.set_ylabel("90 % CIs", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("total_sims", fontsize=10)

    # add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        fontsize=7,
    )
    plt.subplots_adjust(wspace=0.18, hspace=0.22)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    fig.suptitle(r"OU process ($\mu=0.1, \sigma=0.2, \tau=0.5$)")
    plt.show()

if __name__ == "__main__":
    results, all_metrics, all_particles = load_abc_results(DATA_DIR)
    ci_data = compute_ci_trajectories(results, list(true_params.keys()))

    plot(
        ci_data=ci_data,
        true_params=true_params,
        budget_rows=rows,
        save_path=DATA_DIR / "abc_90ci_comparison.png",
    )