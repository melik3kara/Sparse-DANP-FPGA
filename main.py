from __future__ import annotations

import csv
import json
import os
import time
import argparse
from pathlib import Path

import tensorflow as tf

from utils import (
    algorithm_to_flags,
    evaluate_model,
    load_dataset,
    make_loss_fn,
    save_experiment_results,
    set_seed,
)
from models import MLP
from algorithms import optimizer_from_name, train_step, COMPARE_DIAG_KEYS
from analysis import GD_STAT_KEYS, empty_grad_dist_history, save_gradient_dist_results
from sparse import allocate_fractions
from coverage import (
    empty_coverage_history,
    update_coverage_history,
    mean_coverage_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lean implementation of BP/NP/ANP/INP and decorrelated variants."
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"])
    parser.add_argument("--algorithm", type=str, default="danp",
                        choices=["bp", "dbp", "np", "dnp", "anp", "danp", "inp", "dinp"])
    parser.add_argument("--loss", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 1024, 1024])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decor_lr", type=float, default=1e-3)
    parser.add_argument(
        "--decor_update_every", type=int, default=1,
        help=(
            "DANP only: update the decorrelation matrices R only every K global "
            "training steps (lazy decorrelation). R is still APPLIED in the "
            "forward/update path every step; only the R-update itself is skipped on "
            "non-update steps. K=1 (default) updates R every step, identical to prior "
            "behaviour. Has no effect for algorithm=anp (accepted but ignored)."
        ),
    )
    parser.add_argument("--noise_std", type=float, default=1e-2)
    parser.add_argument("--num_noise_iters", type=int, default=1)
    parser.add_argument(
        "--sparse", action="store_true",
        help="Enable sparse/adaptive node perturbation for np/anp/dnp/danp.",
    )
    parser.add_argument(
        "--sparse_fraction", type=float, default=0.5,
        help="Active fraction k/H per layer when --sparse is set.",
    )
    parser.add_argument(
        "--sparse_policy", type=str, default="random",
        choices=["random", "scheduled", "activation_threshold",
                 "activity_diff_topk", "activity_loss_topk", "gradient_aligned_topk"],
        help="Policy for selecting active nodes when --sparse is set.",
    )
    # --- new: layer allocation ---
    parser.add_argument(
        "--layer_allocation", type=str, default="uniform",
        choices=["uniform", "front_loaded", "back_loaded", "middle_loaded"],
        help=(
            "How to distribute the sparse budget across layers while keeping "
            "the total active-node count ≈ sparse_fraction * total_nodes. "
            "Only applies to one-pass policies (random, scheduled, activation_threshold). "
            "'uniform' (default) replicates the existing per-layer behaviour."
        ),
    )
    # --- new: per-layer probe ---
    parser.add_argument(
        "--probe_layer", type=str, default=None,
        help=(
            "Perturb only one layer at a time to measure its individual value. "
            "Pass an integer index (0-based, counting all layers incl. output) "
            "or 'all' to run one complete experiment per layer automatically."
        ),
    )
    # --- new: coverage logging ---
    parser.add_argument(
        "--log_coverage", action="store_true",
        help=(
            "Log per-layer coverage metrics each epoch: mask coverage, "
            "effective activity coverage, mean ||δa_l||², and mean |δa_l|. "
            "Saved to <exp_dir>/coverage.json."
        ),
    )
    parser.add_argument(
        "--coverage_threshold", type=float, default=1e-12,
        help="Minimum mean |δa| for a node to count as effectively active.",
    )
    # --- noise sampling mode ---
    parser.add_argument(
        "--noise_sampling", type=str, default="single",
        choices=["single", "antithetic", "resample"],
        help=(
            "How to draw the perturbation noise each step. "
            "'single' (default): one noisy pass with noise v. "
            "'antithetic': two passes with +v and −v (same mask, same nodes); "
            "  cancels the even-order term in the Taylor expansion of δL. "
            "'resample': two passes with independent v1, v2 on the same mask; "
            "  matched-cost control for antithetic."
        ),
    )
    # --- noise distribution ---
    parser.add_argument(
        "--noise_distribution", type=str, default="gaussian",
        choices=["gaussian", "rademacher"],
        help=(
            "Distribution used to sample node perturbations. "
            "'gaussian' (default): N(0, noise_std²) per node. "
            "'rademacher': ±noise_std with equal probability — same zero mean "
            "and per-node variance as Gaussian, cheaper to generate in hardware."
        ),
    )
    # --- normalizer ablation ---
    parser.add_argument(
        "--norm_mode", type=str, default="exact",
        choices=["exact", "ema", "none"],
        help=(
            "How to compute the ‖δa‖² normalizer in the ANP update. "
            "'exact' (default): per-step per-sample norm — current behaviour, no change. "
            "'ema': exponential moving average of the norm, refreshed every "
            "norm_update_every steps; amortises the global reduction in hardware. "
            "'none': remove the normalizer entirely (scale = δL); diagnostic only."
        ),
    )
    parser.add_argument(
        "--norm_beta", type=float, default=0.99,
        help="EMA decay for --norm_mode ema. Default 0.99.",
    )
    parser.add_argument(
        "--norm_update_every", type=int, default=1,
        help=(
            "Recompute ‖δa‖² every this many training steps for EMA update. "
            "K=1 means every step (like exact but filtered); K>1 amortises cost."
        ),
    )
    parser.add_argument(
        "--norm_eps", type=float, default=1e-8,
        help="Epsilon added to EMA norm denominator for numerical safety.",
    )
    # --- linearized ANP ---
    parser.add_argument(
        "--delta_mode", type=str, default="noisy",
        choices=["noisy", "linearized", "compare"],
        help=(
            "How to compute per-layer activity differences δz_l used in the ANP update. "
            "'noisy' (default): full nonlinear noisy forward pass — current behaviour. "
            "'linearized': first-order approximation "
            "delta_z_l = delta_a_{l-1} @ W_l + epsilon_l (no noisy forward pass). "
            "'compare': run both paths with the same epsilon tensors, train on the noisy "
            "path, and log approximation-quality diagnostics per batch."
        ),
    )
    # --- existing ---
    parser.add_argument(
        "--analyze_gradient_dist", action="store_true",
        help="Collect per-epoch ANP effective-error concentration stats.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--write_results_dir", type=str, default="results")
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment subfolder name. Defaults to dataset_algorithm.",
    )
    parser.add_argument("--save_json", action="store_true")
    return parser.parse_args()


def build_model(info: dict, hidden_sizes: list[int]) -> MLP:
    return MLP(
        input_dim=info["input_dim"],
        hidden_sizes=hidden_sizes,
        output_dim=info["num_classes"],
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.softmax,
    )


def run_single_seed(
    args,
    seed: int,
    probe_layer: int | None = None,   # overrides args.probe_layer when not None
) -> tuple:
    """Train for one seed and return (history, grad_dist_history, runtime_s)."""
    t0 = time.time()
    set_seed(seed)

    base_algorithm, decorrelated = algorithm_to_flags(args.algorithm)
    loss_fn = make_loss_fn(args.loss)

    sparse_active = args.sparse and base_algorithm in {"np", "anp"}

    train_ds, test_ds, info = load_dataset(
        name=args.dataset, batch_size=args.batch_size, flatten=True,
    )

    model = build_model(info=info, hidden_sizes=args.hidden_sizes)
    optimizer = optimizer_from_name(args.optimizer, args.lr)

    # --- Compute per-layer fractions for layer allocation ---
    layer_sizes = [layer.units for layer in model.layers_list]
    n_layers = len(layer_sizes)

    layer_fractions = None
    if sparse_active and args.layer_allocation != "uniform":
        layer_fractions = allocate_fractions(
            layer_sizes=layer_sizes,
            total_fraction=args.sparse_fraction,
            mode=args.layer_allocation,
        )
        print(f"Layer allocation ({args.layer_allocation}): "
              f"{[f'{f:.3f}' for f in layer_fractions]}")

    # --- Resolve probe_layer ---
    effective_probe = probe_layer  # integer or None
    if effective_probe is None and args.probe_layer is not None and args.probe_layer != "all":
        effective_probe = int(args.probe_layer)

    # --- History containers ---
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [],  "test_acc": [],
        "sparse_active_nodes": [], "sparse_total_nodes": [],
        "sparse_relative_cost": [],
        "sparse_score_selected": [], "sparse_score_unselected": [],
    }
    grad_dist_history = empty_grad_dist_history() if args.analyze_gradient_dist else None
    cov_history = empty_coverage_history(n_layers) if args.log_coverage else None

    print(f"\nRunning seed {seed}")
    print(f"Input dim: {info['input_dim']}, classes: {info['num_classes']}")
    if sparse_active:
        print(
            f"Sparse mode: fraction={args.sparse_fraction} | "
            f"policy={args.sparse_policy} | allocation={args.layer_allocation}"
        )
    if effective_probe is not None:
        print(f"Probe mode: only layer {effective_probe} is perturbed")

    # --- Epoch 0: untrained baseline ---
    initial_stats = evaluate_model(model=model, dataset=test_ds,
                                   loss_fn=loss_fn, decorrelated=decorrelated)
    history["train_loss"].append(float("nan"))
    history["train_acc"].append(float("nan"))
    history["test_loss"].append(initial_stats["loss"])
    history["test_acc"].append(initial_stats["acc"])
    history["sparse_active_nodes"].append(float("nan"))
    history["sparse_total_nodes"].append(float("nan"))
    history["sparse_relative_cost"].append(float("nan"))
    history["sparse_score_selected"].append(float("nan"))
    history["sparse_score_unselected"].append(float("nan"))
    if grad_dist_history is not None:
        for k in GD_STAT_KEYS:
            grad_dist_history[k].append(float("nan"))

    print(
        f"Epoch 000/{args.epochs} | "
        f"test loss {initial_stats['loss']:.4f} | "
        f"test acc {initial_stats['acc']:.4f}"
    )

    global_step = 0  # monotonic across epochs; not reset per epoch
    norm_state: dict = {"ema_norm": None}  # EMA of ‖δa‖² for norm_mode="ema"
    decor_updates_performed = 0
    total_train_steps = 0

    for epoch in range(args.epochs):
        train_loss_m  = tf.keras.metrics.Mean()
        train_acc_m   = tf.keras.metrics.CategoricalAccuracy()
        sparse_act_m  = tf.keras.metrics.Mean()
        sparse_tot_m  = tf.keras.metrics.Mean()
        sparse_cost_m = tf.keras.metrics.Mean()
        score_sel_m   = tf.keras.metrics.Mean()
        score_uns_m   = tf.keras.metrics.Mean()
        gd_metrics    = {k: tf.keras.metrics.Mean() for k in GD_STAT_KEYS} \
                        if args.analyze_gradient_dist else None
        cd_metrics    = {k: tf.keras.metrics.Mean() for k in COMPARE_DIAG_KEYS} \
                        if args.delta_mode == "compare" else None
        batch_cov_list: list[list[dict]] = []

        for x, y in train_ds:
            y_pred, loss_per_sample, sparse_info, grad_dist, coverage_stats, compare_diag, \
                decor_update_performed = train_step(
                model=model,
                optimizer=optimizer,
                x=x,
                y=y,
                loss_fn=loss_fn,
                algorithm=base_algorithm,
                decorrelated=decorrelated,
                noise_std=args.noise_std,
                num_noise_iters=args.num_noise_iters,
                decor_lr=args.decor_lr,
                sparse=sparse_active,
                sparse_fraction=args.sparse_fraction,
                sparse_policy=args.sparse_policy,
                step=global_step,
                analyze=args.analyze_gradient_dist,
                layer_fractions=layer_fractions,
                probe_layer=effective_probe,
                log_coverage=args.log_coverage,
                coverage_threshold=args.coverage_threshold,
                noise_distribution=args.noise_distribution,
                noise_sampling=args.noise_sampling,
                norm_mode=args.norm_mode,
                norm_beta=args.norm_beta,
                norm_update_every=args.norm_update_every,
                norm_eps=args.norm_eps,
                norm_state=norm_state,
                delta_mode=args.delta_mode,
                decor_update_every=args.decor_update_every,
            )
            train_loss_m.update_state(tf.reduce_mean(loss_per_sample))
            train_acc_m.update_state(y, y_pred)

            total_train_steps += 1
            if decor_update_performed:
                decor_updates_performed += 1

            if sparse_info is not None:
                sparse_act_m.update_state(sparse_info["n_active"])
                sparse_tot_m.update_state(sparse_info["n_total"])
                sparse_cost_m.update_state(sparse_info["fraction"])
                sel = sparse_info.get("score_selected_mean", float("nan"))
                unsel = sparse_info.get("score_unselected_mean", float("nan"))
                if sel == sel:
                    score_sel_m.update_state(sel)
                if unsel == unsel:
                    score_uns_m.update_state(unsel)

            if gd_metrics is not None and grad_dist:
                for k in GD_STAT_KEYS:
                    v = grad_dist.get(k, float("nan"))
                    if v == v:
                        gd_metrics[k].update_state(v)

            if cd_metrics is not None and compare_diag is not None:
                for k in COMPARE_DIAG_KEYS:
                    v = compare_diag.get(k, float("nan"))
                    if v == v:
                        cd_metrics[k].update_state(v)

            if coverage_stats is not None:
                batch_cov_list.append(coverage_stats)

            global_step += 1

        train_stats = {
            "loss": float(train_loss_m.result().numpy()),
            "acc":  float(train_acc_m.result().numpy()),
        }
        test_stats = evaluate_model(model=model, dataset=test_ds,
                                    loss_fn=loss_fn, decorrelated=decorrelated)

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["test_loss"].append(test_stats["loss"])
        history["test_acc"].append(test_stats["acc"])

        if sparse_active or effective_probe is not None:
            sparse_stats = {
                "active_nodes":    float(sparse_act_m.result().numpy()),
                "total_nodes":     float(sparse_tot_m.result().numpy()),
                "relative_cost":   float(sparse_cost_m.result().numpy()),
                "score_selected":  float(score_sel_m.result().numpy())
                                   if score_sel_m.count.numpy() > 0 else float("nan"),
                "score_unselected": float(score_uns_m.result().numpy())
                                   if score_uns_m.count.numpy() > 0 else float("nan"),
            }
        else:
            sparse_stats = {k: float("nan") for k in
                            ("active_nodes", "total_nodes", "relative_cost",
                             "score_selected", "score_unselected")}

        history["sparse_active_nodes"].append(sparse_stats["active_nodes"])
        history["sparse_total_nodes"].append(sparse_stats["total_nodes"])
        history["sparse_relative_cost"].append(sparse_stats["relative_cost"])
        history["sparse_score_selected"].append(sparse_stats["score_selected"])
        history["sparse_score_unselected"].append(sparse_stats["score_unselected"])

        if grad_dist_history is not None:
            epoch_gd = {
                k: float(gd_metrics[k].result().numpy())
                   if gd_metrics[k].count.numpy() > 0 else float("nan")
                for k in GD_STAT_KEYS
            }
            for k in GD_STAT_KEYS:
                grad_dist_history[k].append(epoch_gd[k])
        else:
            epoch_gd = None

        if cov_history is not None and batch_cov_list:
            epoch_cov_means = mean_coverage_stats(batch_cov_list)
            update_coverage_history(cov_history, epoch_cov_means)

        log_line = (
            f"Seed {seed} | Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_stats['loss']:.4f} | train acc {train_stats['acc']:.4f} | "
            f"test loss {test_stats['loss']:.4f} | test acc {test_stats['acc']:.4f}"
        )
        if sparse_active or effective_probe is not None:
            log_line += (
                f" | active {sparse_stats['active_nodes']:.1f}/"
                f"{sparse_stats['total_nodes']:.0f} "
                f"cost={sparse_stats['relative_cost']:.3f}"
            )
        if effective_probe is not None:
            log_line += f" [probe_l{effective_probe}]"
        print(log_line)
        if epoch_gd is not None:
            cv_s  = f"{epoch_gd['g_cv']:.3f}"      if epoch_gd['g_cv']     == epoch_gd['g_cv']     else "nan"
            t10_s = f"{epoch_gd['top_10pct']:.3f}"  if epoch_gd['top_10pct'] == epoch_gd['top_10pct'] else "nan"
            gi_s  = f"{epoch_gd['gini']:.3f}"       if epoch_gd['gini']      == epoch_gd['gini']      else "nan"
            print(f"  grad_dist: CV={cv_s}  top10%={t10_s}  gini={gi_s}")

        if cd_metrics is not None:
            epoch_cd = {
                k: float(cd_metrics[k].result().numpy())
                   if cd_metrics[k].count.numpy() > 0 else float("nan")
                for k in COMPARE_DIAG_KEYS
            }
            print(
                f"  compare: rel_δz={epoch_cd['rel_delta_z_error']:.4f}  "
                f"rel_δlogits={epoch_cd['rel_delta_logits_error']:.4f}  "
                f"rel_δL={epoch_cd['rel_delta_L_error']:.4f}"
            )
        else:
            epoch_cd = None

    runtime_s = time.time() - t0
    ema_norm_final = norm_state.get("ema_norm")  # None when norm_mode != "ema"

    # Final compare diagnostics (last epoch mean, or None)
    compare_final: dict | None = None
    if epoch_cd is not None:
        hidden_units = sum(l.units for l in model.layers_list[:-1])
        compare_final = {
            **epoch_cd,
            "full_noisy_forward_used":               1,
            "linearized_delta_path_used":            1,
            "nonlinear_noisy_activation_evals_saved": 0,
        }
    elif args.delta_mode == "linearized":
        hidden_units = sum(l.units for l in model.layers_list[:-1])
        compare_final = {
            **{k: float("nan") for k in COMPARE_DIAG_KEYS},
            "full_noisy_forward_used":               0,
            "linearized_delta_path_used":            1,
            "nonlinear_noisy_activation_evals_saved": args.batch_size * hidden_units,
        }

    decor_update_fraction = (
        decor_updates_performed / total_train_steps if total_train_steps > 0 else float("nan")
    )
    decor_stats = {
        "decor_update_every":       args.decor_update_every,
        "decor_updates_performed":  decor_updates_performed,
        "total_train_steps":        total_train_steps,
        "decor_update_fraction":    decor_update_fraction,
    }

    print(f"\nFinished seed {seed} in {runtime_s:.1f}s | "
          f"final test acc {history['test_acc'][-1]:.4f} | "
          f"best test acc {max(a for a in history['test_acc'] if a == a):.4f}")
    return history, grad_dist_history, cov_history, runtime_s, ema_norm_final, compare_final, decor_stats


def save_run_summary_csv(
    result_dir: Path,
    seed: int,
    history: dict,
    config: dict,
    runtime_s: float,
) -> None:
    """Append one row to <result_dir>/run_summary.csv."""
    valid_test_acc = [a for a in history["test_acc"] if a == a]
    best_test_acc  = max(valid_test_acc) if valid_test_acc else float("nan")
    best_epoch     = (history["test_acc"].index(best_test_acc)
                      if best_test_acc in history["test_acc"] else -1)

    row = {
        "exp_name":            result_dir.name,
        "algorithm":           config.get("algorithm", ""),
        "dataset":             config.get("dataset", ""),
        "hidden_sizes":        str(config.get("hidden_sizes", "")),
        "sparse":              config.get("sparse", False),
        "sparse_fraction":     config.get("sparse_fraction", float("nan")),
        "sparse_policy":       config.get("sparse_policy", ""),
        "layer_allocation":    config.get("layer_allocation", "uniform"),
        "probe_layer":         config.get("probe_layer", ""),
        "noise_distribution":  config.get("noise_distribution", "gaussian"),
        "noise_sampling":      config.get("noise_sampling", "single"),
        "noisy_passes":        1 if config.get("noise_sampling", "single") == "single" else 2,
        "norm_mode":           config.get("norm_mode", "exact"),
        "norm_beta":           config.get("norm_beta", 0.99),
        "norm_update_every":   config.get("norm_update_every", 1),
        "norm_eps":            config.get("norm_eps", 1e-8),
        "ema_norm_final":      config.get("ema_norm_final", float("nan")),
        "decor_update_every":      config.get("decor_update_every", 1),
        "decor_updates_performed": config.get("decor_updates_performed", 0),
        "total_train_steps":       config.get("total_train_steps", 0),
        "decor_update_fraction":   config.get("decor_update_fraction", float("nan")),
        "delta_mode":          config.get("delta_mode", "noisy"),
        "full_noisy_forward_used":
            config.get("full_noisy_forward_used",
                       1 if config.get("delta_mode", "noisy") in {"noisy", "compare"} else 0),
        "linearized_delta_path_used":
            config.get("linearized_delta_path_used",
                       1 if config.get("delta_mode", "noisy") in {"linearized", "compare"} else 0),
        "nonlinear_noisy_activation_evals_saved":
            config.get("nonlinear_noisy_activation_evals_saved", 0),
        **{f"compare_{k}": config.get(k, float("nan")) for k in COMPARE_DIAG_KEYS},
        "lr":                  config.get("lr", float("nan")),
        "epochs":              config.get("epochs", ""),
        "seed":                seed,
        "train_acc_final":     history["train_acc"][-1],
        "test_acc_final":      history["test_acc"][-1],
        "train_loss_final":    history["train_loss"][-1],
        "test_loss_final":     history["test_loss"][-1],
        "best_test_acc":       best_test_acc,
        "best_test_acc_epoch": best_epoch,
        "rel_cost_final":      history["sparse_relative_cost"][-1],
        "runtime_s":           round(runtime_s, 2),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "run_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_coverage_json(result_dir: Path, per_seed_cov: list, config: dict) -> None:
    """Save per-seed per-epoch coverage stats to <result_dir>/coverage.json."""
    import json, numpy as np

    def _clean(v):
        return None if (v is None or v != v) else float(v)

    seeds_data = []
    for seed, cov_hist in per_seed_cov:
        seeds_data.append({"seed": seed, "history": {k: [_clean(x) for x in v]
                                                      for k, v in cov_hist.items()}})
    with open(result_dir / "coverage.json", "w") as f:
        json.dump({"config": config, "seeds": seeds_data}, f, indent=2)
    print(f"Coverage stats saved to {result_dir / 'coverage.json'}")


# ---------------------------------------------------------------------------
# Probe-all runner
# ---------------------------------------------------------------------------

def run_probe_all(args) -> None:
    """Run one complete experiment per layer and save a combined probe CSV."""
    base_algorithm, _ = algorithm_to_flags(args.algorithm)
    # Build a temporary model just to count layers.
    train_ds, test_ds, info = load_dataset(
        name=args.dataset, batch_size=args.batch_size, flatten=True,
    )
    model = build_model(info=info, hidden_sizes=args.hidden_sizes)
    n_layers = len(model.layers_list)
    layer_sizes = [l.units for l in model.layers_list]
    del model

    base_exp_name = args.exp_name or f"{args.dataset}_{args.algorithm}"
    probe_rows: list[dict] = []

    for l_idx in range(n_layers):
        exp_name = f"{base_exp_name}_probe_l{l_idx}"
        result_dir = Path(args.write_results_dir) / exp_name
        print(f"\n{'='*60}")
        print(f"PROBE LAYER {l_idx}/{n_layers-1}  (size={layer_sizes[l_idx]})  → {result_dir}")
        print(f"{'='*60}")

        all_histories: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "test_loss": [],  "test_acc": [],
            "sparse_active_nodes": [], "sparse_total_nodes": [],
            "sparse_relative_cost": [],
            "sparse_score_selected": [], "sparse_score_unselected": [],
        }

        config = vars(args).copy()
        config["probe_layer"] = l_idx

        for seed_offset in range(args.num_seeds):
            seed = args.seed + seed_offset
            history, grad_dist_history, cov_history, runtime_s, _ema, _cmp, _decor = run_single_seed(
                args, seed, probe_layer=l_idx,
            )
            for key in all_histories:
                all_histories[key].append(history[key])

            valid = [a for a in history["test_acc"] if a == a]
            best  = max(valid) if valid else float("nan")
            probe_rows.append({
                "layer_idx":    l_idx,
                "layer_size":   layer_sizes[l_idx],
                "seed":         seed,
                "test_acc_final": history["test_acc"][-1],
                "best_test_acc":  best,
                "runtime_s":      round(runtime_s, 2),
            })
            save_run_summary_csv(result_dir, seed=seed, history=history,
                                 config=config, runtime_s=runtime_s)

        save_experiment_results(
            write_results_dir=args.write_results_dir,
            exp_name=exp_name,
            histories=all_histories,
            config=config,
            per_seed_payload=None,
        )

    # Combined probe CSV
    probe_csv_path = Path(args.write_results_dir) / f"{base_exp_name}_probe_summary.csv"
    if probe_rows:
        with open(probe_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
            writer.writeheader()
            writer.writerows(probe_rows)
    print(f"\nProbe summary saved to {probe_csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Configuration")
    print(json.dumps(vars(args), indent=2))

    # Handle probe_layer == "all" separately.
    if args.probe_layer == "all":
        run_probe_all(args)
        return

    exp_name = args.exp_name if args.exp_name is not None else f"{args.dataset}_{args.algorithm}"

    all_histories: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "test_loss": [],  "test_acc": [],
        "sparse_active_nodes": [], "sparse_total_nodes": [],
        "sparse_relative_cost": [],
        "sparse_score_selected": [], "sparse_score_unselected": [],
    }
    per_seed_payload: list = []
    per_seed_gd: list = []
    per_seed_cov: list = []

    result_dir = Path(args.write_results_dir) / exp_name

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        history, grad_dist_history, cov_history, runtime_s, ema_norm_final, compare_final, decor_stats = \
            run_single_seed(args, seed)

        for key in all_histories:
            all_histories[key].append(history[key])

        per_seed_payload.append({"seed": seed, "history": history})

        if args.analyze_gradient_dist and grad_dist_history is not None:
            per_seed_gd.append((seed, grad_dist_history))

        if args.log_coverage and cov_history is not None:
            per_seed_cov.append((seed, cov_history))

        # Per-seed CSV row
        csv_config = {**vars(args), "ema_norm_final": ema_norm_final}
        if compare_final is not None:
            csv_config.update(compare_final)
        if decor_stats is not None:
            csv_config.update(decor_stats)
        save_run_summary_csv(result_dir, seed=seed, history=history,
                             config=csv_config, runtime_s=runtime_s)

    save_experiment_results(
        write_results_dir=args.write_results_dir,
        exp_name=exp_name,
        histories=all_histories,
        config=vars(args),
        per_seed_payload=per_seed_payload if args.save_json else None,
    )

    if args.analyze_gradient_dist and per_seed_gd:
        save_gradient_dist_results(result_dir, per_seed_gd, vars(args))

    if args.log_coverage and per_seed_cov:
        save_coverage_json(result_dir, per_seed_cov, vars(args))

    print(f"\nSaved results to {result_dir}")


if __name__ == "__main__":
    main()
