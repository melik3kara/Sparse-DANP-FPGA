"""Budget sweep: active_fraction × policy × algorithm × seed → combined CSV.

Runs a grid of experiments and writes one row per (fraction, policy, algorithm, seed)
to results/sparse_anp_experiments/budget_sweep.csv.

Usage
-----
    python sweep_budget.py --dataset mnist --hidden_sizes 1024 1024 1024 --epochs 50
    python sweep_budget.py --dataset mnist --epochs 50 --fractions 0.0625 0.125 0.25 0.5

All main.py flags that are not part of the sweep grid are forwarded as-is.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path


FRACTIONS_DEFAULT  = [0.0625, 0.125, 0.25, 0.5]
POLICIES_DEFAULT   = ["random", "scheduled"]
ALGORITHMS_DEFAULT = ["anp", "danp"]
SEEDS_DEFAULT      = [42, 43, 44]

RESULTS_ROOT = Path("results/sparse_anp_experiments")
SUMMARY_CSV  = RESULTS_ROOT / "budget_sweep.csv"

CSV_FIELDS = [
    "algorithm", "sparse_fraction", "sparse_policy",
    "seed", "dataset", "hidden_sizes", "lr", "epochs",
    "train_acc_final", "test_acc_final",
    "train_loss_final", "test_loss_final",
    "best_test_acc", "best_test_acc_epoch",
    "rel_cost_final", "runtime_s",
]


def parse_args():
    p = argparse.ArgumentParser(description="Budget sweep driver.")
    p.add_argument("--dataset",      default="mnist")
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=1000)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--noise_std",    type=float, default=1e-2)
    p.add_argument("--loss",         default="ce")
    p.add_argument("--optimizer",    default="adam")
    p.add_argument("--num_noise_iters", type=int, default=1)
    p.add_argument("--layer_allocation", default="uniform",
                   choices=["uniform","front_loaded","back_loaded","middle_loaded"])
    # sweep dimensions
    p.add_argument("--fractions",   type=float, nargs="+", default=FRACTIONS_DEFAULT)
    p.add_argument("--policies",    type=str,   nargs="+", default=POLICIES_DEFAULT)
    p.add_argument("--algorithms",  type=str,   nargs="+", default=ALGORITHMS_DEFAULT)
    p.add_argument("--seeds",       type=int,   nargs="+", default=SEEDS_DEFAULT)
    # dense baseline
    p.add_argument("--include_dense_baseline", action="store_true",
                   help="Also run full ANP/DANP with sparse disabled as a reference.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument("--gpu", type=str, default=None)
    return p.parse_args()


def _build_main_cmd(
    *,
    algorithm: str,
    sparse_fraction: float,
    sparse_policy: str,
    seed: int,
    args,
    exp_name: str,
    sparse: bool = True,
) -> list[str]:
    cmd = [
        sys.executable, "main.py",
        "--algorithm",      algorithm,
        "--dataset",        args.dataset,
        "--hidden_sizes",   *[str(h) for h in args.hidden_sizes],
        "--epochs",         str(args.epochs),
        "--batch_size",     str(args.batch_size),
        "--lr",             str(args.lr),
        "--noise_std",      str(args.noise_std),
        "--loss",           args.loss,
        "--optimizer",      args.optimizer,
        "--num_noise_iters", str(args.num_noise_iters),
        "--seed",           str(seed),
        "--num_seeds",      "1",
        "--write_results_dir", str(RESULTS_ROOT),
        "--exp_name",       exp_name,
        "--layer_allocation", args.layer_allocation,
    ]
    if sparse:
        cmd += ["--sparse", "--sparse_fraction", str(sparse_fraction),
                "--sparse_policy", sparse_policy]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    return cmd


def _read_run_summary_csv(result_dir: Path) -> dict | None:
    """Read the last row of run_summary.csv and return it as a dict."""
    csv_path = result_dir / "run_summary.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def run_sweep(args):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    summary_file = open(SUMMARY_CSV, "a", newline="")
    writer = csv.DictWriter(summary_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    grid = list(itertools.product(args.fractions, args.policies, args.algorithms, args.seeds))
    if args.include_dense_baseline:
        dense_grid = list(itertools.product(args.algorithms, args.seeds))
    else:
        dense_grid = []

    total = len(grid) + len(dense_grid)
    done  = 0

    # --- Dense baselines ---
    for alg, seed in dense_grid:
        exp_name = f"budget_dense_{alg}_s{seed}"
        result_dir = RESULTS_ROOT / exp_name
        cmd = _build_main_cmd(
            algorithm=alg,
            sparse_fraction=1.0,
            sparse_policy="random",
            seed=seed,
            args=args,
            exp_name=exp_name,
            sparse=False,
        )
        done += 1
        print(f"\n[{done}/{total}] DENSE  alg={alg}  seed={seed}")
        if args.dry_run:
            print("  DRY_RUN:", " ".join(cmd))
            continue
        t0 = time.time()
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        row_data = _read_run_summary_csv(result_dir) or {}
        row = {
            "algorithm": alg, "sparse_fraction": 1.0, "sparse_policy": "none",
            "seed": seed, "dataset": args.dataset,
            "hidden_sizes": str(args.hidden_sizes), "lr": args.lr,
            "epochs": args.epochs, "runtime_s": round(elapsed, 2),
        }
        row.update({k: row_data.get(k, "") for k in CSV_FIELDS if k not in row})
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        summary_file.flush()

    # --- Sparse grid ---
    for frac, policy, alg, seed in grid:
        exp_name = f"budget_f{frac}_p{policy}_{alg}_s{seed}"
        result_dir = RESULTS_ROOT / exp_name
        cmd = _build_main_cmd(
            algorithm=alg,
            sparse_fraction=frac,
            sparse_policy=policy,
            seed=seed,
            args=args,
            exp_name=exp_name,
            sparse=True,
        )
        done += 1
        print(f"\n[{done}/{total}] frac={frac}  policy={policy}  alg={alg}  seed={seed}")
        if args.dry_run:
            print("  DRY_RUN:", " ".join(cmd))
            continue
        t0 = time.time()
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        row_data = _read_run_summary_csv(result_dir) or {}
        row = {
            "algorithm": alg, "sparse_fraction": frac, "sparse_policy": policy,
            "seed": seed, "dataset": args.dataset,
            "hidden_sizes": str(args.hidden_sizes), "lr": args.lr,
            "epochs": args.epochs, "runtime_s": round(elapsed, 2),
        }
        row.update({k: row_data.get(k, "") for k in CSV_FIELDS if k not in row})
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        summary_file.flush()

    summary_file.close()
    print(f"\nBudget sweep complete. Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    run_sweep(parse_args())
