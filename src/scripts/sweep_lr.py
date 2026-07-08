"""LR sweep: learning_rate × active_fraction × seed → combined CSV.

Runs a grid of experiments across LR values for each active fraction and writes
one row per (lr, fraction, algorithm, seed) to:
    results/sparse_anp_experiments/lr_sweep.csv

Usage
-----
    python sweep_lr.py --dataset mnist --epochs 50 --algorithm anp
    python sweep_lr.py --dataset mnist --epochs 30 \\
        --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 --fractions 0.125 0.25 0.5
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path
import subprocess


LRS_DEFAULT       = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
FRACTIONS_DEFAULT = [0.125, 0.25, 0.5]
SEEDS_DEFAULT     = [42, 43, 44]

RESULTS_ROOT = Path("results/sparse_anp_experiments")
SUMMARY_CSV  = RESULTS_ROOT / "lr_sweep.csv"

CSV_FIELDS = [
    "algorithm", "lr", "sparse_fraction", "sparse_policy",
    "seed", "dataset", "hidden_sizes", "epochs",
    "train_acc_final", "test_acc_final",
    "train_loss_final", "test_loss_final",
    "best_test_acc", "best_test_acc_epoch",
    "rel_cost_final", "runtime_s",
]


def parse_args():
    p = argparse.ArgumentParser(description="LR sweep driver.")
    p.add_argument("--dataset",       default="mnist")
    p.add_argument("--algorithm",     default="anp",
                   choices=["anp","danp","np","dnp"])
    p.add_argument("--hidden_sizes",  type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=1000)
    p.add_argument("--noise_std",     type=float, default=1e-2)
    p.add_argument("--loss",          default="ce")
    p.add_argument("--optimizer",     default="adam")
    p.add_argument("--sparse_policy", default="scheduled",
                   choices=["random","scheduled","activation_threshold"])
    p.add_argument("--layer_allocation", default="uniform",
                   choices=["uniform","front_loaded","back_loaded","middle_loaded"])
    # sweep dimensions
    p.add_argument("--lrs",      type=float, nargs="+", default=LRS_DEFAULT)
    p.add_argument("--fractions",type=float, nargs="+", default=FRACTIONS_DEFAULT)
    p.add_argument("--seeds",    type=int,   nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--dry_run",  action="store_true")
    p.add_argument("--gpu",      type=str,   default=None)
    return p.parse_args()


def _build_cmd(*, lr: float, fraction: float, seed: int, args, exp_name: str) -> list[str]:
    cmd = [
        sys.executable, "main.py",
        "--algorithm",      args.algorithm,
        "--dataset",        args.dataset,
        "--hidden_sizes",   *[str(h) for h in args.hidden_sizes],
        "--epochs",         str(args.epochs),
        "--batch_size",     str(args.batch_size),
        "--lr",             str(lr),
        "--noise_std",      str(args.noise_std),
        "--loss",           args.loss,
        "--optimizer",      args.optimizer,
        "--seed",           str(seed),
        "--num_seeds",      "1",
        "--write_results_dir", str(RESULTS_ROOT),
        "--exp_name",       exp_name,
        "--sparse",
        "--sparse_fraction", str(fraction),
        "--sparse_policy",   args.sparse_policy,
        "--layer_allocation", args.layer_allocation,
    ]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    return cmd


def _read_last_row(result_dir: Path) -> dict:
    csv_path = result_dir / "run_summary.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def run_sweep(args):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    out_file = open(SUMMARY_CSV, "a", newline="")
    writer = csv.DictWriter(out_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    grid = list(itertools.product(args.lrs, args.fractions, args.seeds))
    total = len(grid)

    for done, (lr, frac, seed) in enumerate(grid, 1):
        lr_str   = f"{lr:.0e}".replace("-0", "-")
        exp_name = f"lr{lr_str}_f{frac}_{args.algorithm}_s{seed}"
        result_dir = RESULTS_ROOT / exp_name
        cmd = _build_cmd(lr=lr, fraction=frac, seed=seed, args=args, exp_name=exp_name)

        print(f"\n[{done}/{total}] lr={lr}  frac={frac}  alg={args.algorithm}  seed={seed}")
        if args.dry_run:
            print("  DRY_RUN:", " ".join(cmd))
            continue

        t0 = time.time()
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0

        row_data = _read_last_row(result_dir)
        row = {
            "algorithm":      args.algorithm,
            "lr":             lr,
            "sparse_fraction": frac,
            "sparse_policy":  args.sparse_policy,
            "seed":           seed,
            "dataset":        args.dataset,
            "hidden_sizes":   str(args.hidden_sizes),
            "epochs":         args.epochs,
            "runtime_s":      round(elapsed, 2),
        }
        row.update({k: row_data.get(k, "") for k in CSV_FIELDS if k not in row})
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        out_file.flush()

    out_file.close()
    print(f"\nLR sweep complete. Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    run_sweep(parse_args())
