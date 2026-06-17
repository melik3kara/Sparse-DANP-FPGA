from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def read_metric(exp_dir: Path, metric_name: str, suffix: str) -> np.ndarray:
    path = exp_dir / f"{metric_name}_{suffix}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    values = np.loadtxt(path)
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)
    return values


def infer_label(exp_dir: Path) -> str:
    name = exp_dir.name.lower()
    candidates = ["danp", "dinp", "dnp", "dbp", "anp", "inp", "np", "bp"]
    for candidate in candidates:
        if candidate in name:
            return candidate.upper()
    return exp_dir.name.upper()


def is_decorrelated(label: str) -> bool:
    label = label.lower()
    return label.startswith("d") and label in {"dbp", "dnp", "danp", "dinp"}


def base_algorithm(label: str) -> str:
    label = label.lower()
    if label.startswith("d") and label in {"dbp", "dnp", "danp", "dinp"}:
        return label[1:]
    return label


def get_style(label: str):
    """
    Paper-style color mapping:
    - BP / DBP   : pink
    - NP / DNP   : blue
    - ANP / DANP : green
    - INP / DINP : orange

    Decorrelated variants:
    - solid
    - stronger alpha

    Non-decorrelated variants:
    - same color
    - more transparent
    """
    base = base_algorithm(label)
    decorrelated = is_decorrelated(label)

    color_map = {
        "bp":  "#CC79A7",  # pink
        "np":  "#56B4E9",  # blue
        "anp": "#66C2A5",  # green/teal
        "inp": "#D98C4C",  # orange-brown
    }

    color = color_map.get(base, None)

    if decorrelated:
        line_alpha = 1.0
        fill_alpha = 0.18
        linewidth = 2.2
        linestyle = "-"
    else:
        line_alpha = 0.55
        fill_alpha = 0.12
        linewidth = 2.0
        linestyle = "-"

    return {
        "color": color,
        "line_alpha": line_alpha,
        "fill_alpha": fill_alpha,
        "linewidth": linewidth,
        "linestyle": linestyle,
    }


def get_experiment_dirs(parent_dir: Path) -> list[Path]:
    exp_dirs = [p for p in parent_dir.iterdir() if p.is_dir()]
    order = {"BP": 0, "DBP": 1, "NP": 2, "DNP": 3, "ANP": 4, "DANP": 5, "INP": 6, "DINP": 7}
    exp_dirs.sort(key=lambda p: order.get(infer_label(p), 999))
    return exp_dirs


def plot_metric(
    exp_dirs: list[Path],
    metric_name: str,
    out_path: Path,
    ylabel: str,
    logscale: bool = False,
) -> None:
    plt.figure(figsize=(6.4, 3.2), dpi=200)

    for exp_dir in exp_dirs:
        mean_values = read_metric(exp_dir, metric_name, "mean")
        min_values = read_metric(exp_dir, metric_name, "min")
        max_values = read_metric(exp_dir, metric_name, "max")

        label = infer_label(exp_dir)
        style = get_style(label)
        epochs = np.arange(len(mean_values))

        plt.plot(
            epochs,
            mean_values,
            label=label,
            color=style["color"],
            alpha=style["line_alpha"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
        )
        plt.fill_between(
            epochs,
            min_values,
            max_values,
            color=style["color"],
            alpha=style["fill_alpha"],
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if not logscale:
        ax.set_ylim(bottom=0.0)
    else:
        plt.yscale("log")

    plt.legend(fontsize=8, ncol=4, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot all experiments inside a results folder.")
    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Folder containing experiment subfolders.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Folder where plot PNGs are written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_folder = Path(args.results_folder)
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder does not exist: {results_folder}")

    exp_dirs = get_experiment_dirs(results_folder)
    if not exp_dirs:
        raise ValueError(f"No experiment subfolders found in: {results_folder}")

    folder_tag = results_folder.name
    plots_dir = Path(args.plots_dir)

    plot_metric(
        exp_dirs=exp_dirs,
        metric_name="train_acc",
        out_path=plots_dir / f"{folder_tag}_train_acc.png",
        ylabel="Train accuracy",
        logscale=False,
    )
    plot_metric(
        exp_dirs=exp_dirs,
        metric_name="test_acc",
        out_path=plots_dir / f"{folder_tag}_test_acc.png",
        ylabel="Test accuracy",
        logscale=False,
    )
    plot_metric(
        exp_dirs=exp_dirs,
        metric_name="train_loss",
        out_path=plots_dir / f"{folder_tag}_train_loss.png",
        ylabel="Train loss",
        logscale=True,
    )
    plot_metric(
        exp_dirs=exp_dirs,
        metric_name="test_loss",
        out_path=plots_dir / f"{folder_tag}_test_loss.png",
        ylabel="Test loss",
        logscale=True,
    )

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()