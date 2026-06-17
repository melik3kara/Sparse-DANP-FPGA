from __future__ import annotations

import os
import argparse
import json
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
from algorithms import optimizer_from_name, train_step
from analysis import GD_STAT_KEYS, empty_grad_dist_history, save_gradient_dist_results


def parse_args():
    parser = argparse.ArgumentParser(description="Lean implementation of BP/NP/ANP/INP and decorrelated variants.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default="danp",
        choices=["bp", "dbp", "np", "dnp", "anp", "danp", "inp", "dinp"],
    )
    parser.add_argument("--loss", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 1024, 1024])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decor_lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=1e-2)
    parser.add_argument("--num_noise_iters", type=int, default=1)
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Enable sparse/adaptive node perturbation and update for np/anp/dnp/danp.",
    )
    parser.add_argument(
        "--sparse_fraction",
        type=float,
        default=0.5,
        help="Active fraction k/H of nodes per layer when --sparse is set.",
    )
    parser.add_argument(
        "--sparse_policy",
        type=str,
        default="random",
        choices=["random", "scheduled", "activation_threshold", "activity_diff_topk", "activity_loss_topk", "gradient_aligned_topk"],
        help="Policy for selecting the active nodes per layer when --sparse is set.",
    )
    parser.add_argument(
        "--analyze_gradient_dist",
        action="store_true",
        help=(
            "Collect per-epoch statistics on the ANP effective-error magnitude "
            "|g_i| = |activity_diff_i * perf_diff / network_norm_sq| across all "
            "nodes (CV, top-k energy fractions, Gini). Saved to gradient_dist.json. "
            "Only meaningful for np/anp/dnp/danp algorithms."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds to run.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id to use (e.g. 0 or 1). If not set, TensorFlow chooses automatically.")
    parser.add_argument(
        "--write_results_dir",
        type=str,
        default="results",
        help="Directory where this experiment folder will be written.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
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


def run_single_seed(args, seed: int) -> dict:
    set_seed(seed)

    base_algorithm, decorrelated = algorithm_to_flags(args.algorithm)
    loss_fn = make_loss_fn(args.loss)

    # Sparse perturbation/update is only meaningful for np/anp (and their
    # decorrelated dnp/danp variants). Ignore --sparse for bp/inp so logging
    # doesn't report misleading all-zero sparsity stats.
    sparse_active = args.sparse and base_algorithm in {"np", "anp"}

    train_ds, test_ds, info = load_dataset(
        name=args.dataset,
        batch_size=args.batch_size,
        flatten=True,
    )

    model = build_model(info=info, hidden_sizes=args.hidden_sizes)
    optimizer = optimizer_from_name(args.optimizer, args.lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "sparse_active_nodes": [],
        "sparse_total_nodes": [],
        "sparse_relative_cost": [],
        "sparse_score_selected": [],
        "sparse_score_unselected": [],
    }
    grad_dist_history = empty_grad_dist_history() if args.analyze_gradient_dist else None

    print(f"\nRunning seed {seed}")
    print(f"Input dim: {info['input_dim']}, classes: {info['num_classes']}")
    if sparse_active:
        print(
            f"Sparse mode enabled | fraction={args.sparse_fraction} | "
            f"policy={args.sparse_policy}"
        )

    # --- evaluate untrained model (epoch 0) ---
    initial_stats = evaluate_model(
        model=model,
        dataset=test_ds,
        loss_fn=loss_fn,
        decorrelated=decorrelated,
    )

    history["train_loss"].append(float("nan"))  # no train loss yet
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

    global_step = 0

    for epoch in range(args.epochs):
        train_loss_metric = tf.keras.metrics.Mean()
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        sparse_active_metric = tf.keras.metrics.Mean()
        sparse_total_metric = tf.keras.metrics.Mean()
        sparse_cost_metric = tf.keras.metrics.Mean()
        score_selected_metric = tf.keras.metrics.Mean()
        score_unselected_metric = tf.keras.metrics.Mean()
        gd_metrics = {k: tf.keras.metrics.Mean() for k in GD_STAT_KEYS} if args.analyze_gradient_dist else None

        for x, y in train_ds:
            y_pred, loss_per_sample, sparse_info, grad_dist = train_step(
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
            )
            train_loss_metric.update_state(tf.reduce_mean(loss_per_sample))
            train_acc_metric.update_state(y, y_pred)
            if sparse_info is not None:
                sparse_active_metric.update_state(sparse_info["n_active"])
                sparse_total_metric.update_state(sparse_info["n_total"])
                sparse_cost_metric.update_state(sparse_info["fraction"])
                sel = sparse_info.get("score_selected_mean", float("nan"))
                unsel = sparse_info.get("score_unselected_mean", float("nan"))
                if sel == sel:  # not NaN
                    score_selected_metric.update_state(sel)
                if unsel == unsel:  # not NaN
                    score_unselected_metric.update_state(unsel)
            if gd_metrics is not None and grad_dist:
                for k in GD_STAT_KEYS:
                    v = grad_dist.get(k, float("nan"))
                    if v == v:  # skip NaN
                        gd_metrics[k].update_state(v)
            global_step += 1

        train_stats = {
            "loss": float(train_loss_metric.result().numpy()),
            "acc": float(train_acc_metric.result().numpy()),
        }
        test_stats = evaluate_model(
            model=model,
            dataset=test_ds,
            loss_fn=loss_fn,
            decorrelated=decorrelated,
        )

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["test_loss"].append(test_stats["loss"])
        history["test_acc"].append(test_stats["acc"])

        if sparse_active:
            sparse_stats = {
                "active_nodes": float(sparse_active_metric.result().numpy()),
                "total_nodes": float(sparse_total_metric.result().numpy()),
                "relative_cost": float(sparse_cost_metric.result().numpy()),
                "score_selected": float(score_selected_metric.result().numpy())
                if score_selected_metric.count.numpy() > 0 else float("nan"),
                "score_unselected": float(score_unselected_metric.result().numpy())
                if score_unselected_metric.count.numpy() > 0 else float("nan"),
            }
        else:
            sparse_stats = {
                "active_nodes": float("nan"),
                "total_nodes": float("nan"),
                "relative_cost": float("nan"),
                "score_selected": float("nan"),
                "score_unselected": float("nan"),
            }
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

        log_line = (
            f"Seed {seed} | Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_stats['loss']:.4f} | train acc {train_stats['acc']:.4f} | "
            f"test loss {test_stats['loss']:.4f} | test acc {test_stats['acc']:.4f}"
        )
        if sparse_active:
            log_line += (
                f" | active {sparse_stats['active_nodes']:.1f}/"
                f"{sparse_stats['total_nodes']:.0f} "
                f"[{args.sparse_policy}] "
                f"cost={sparse_stats['relative_cost']:.3f}"
            )
            if args.sparse_policy in {"activity_diff_topk", "activity_loss_topk", "gradient_aligned_topk"}:
                log_line += (
                    f" | score sel={sparse_stats['score_selected']:.4f}"
                    f" unsel={sparse_stats['score_unselected']:.4f}"
                )
        print(log_line)
        if epoch_gd is not None:
            cv_s  = f"{epoch_gd['g_cv']:.3f}"     if epoch_gd['g_cv']     == epoch_gd['g_cv']     else "nan"
            t10_s = f"{epoch_gd['top_10pct']:.3f}" if epoch_gd['top_10pct'] == epoch_gd['top_10pct'] else "nan"
            gi_s  = f"{epoch_gd['gini']:.3f}"      if epoch_gd['gini']      == epoch_gd['gini']      else "nan"
            print(f"  grad_dist: CV={cv_s}  top10%={t10_s}  gini={gi_s}")

    print("\nFinal metrics for seed", seed)
    final_metrics = {k: v[-1] for k, v in history.items()}
    print(json.dumps(final_metrics, indent=2))
    print(f"Final test accuracy: {final_metrics['test_acc']:.4f}")
    if sparse_active:
        print(
            f"Final relative perturbation/update cost [{args.sparse_policy}]: "
            f"{final_metrics['sparse_relative_cost']:.3f} "
            f"({final_metrics['sparse_active_nodes']:.1f}/"
            f"{final_metrics['sparse_total_nodes']:.0f} nodes)"
        )
        if args.sparse_policy in {"activity_diff_topk", "activity_loss_topk", "gradient_aligned_topk"}:
            print(
                f"Final mean score — "
                f"selected: {final_metrics['sparse_score_selected']:.4f} | "
                f"unselected: {final_metrics['sparse_score_unselected']:.4f}"
            )
    return history, grad_dist_history


def main():
    args = parse_args()

    exp_name = args.exp_name if args.exp_name is not None else f"{args.dataset}_{args.algorithm}"

    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Configuration")
    print(json.dumps(vars(args), indent=2))

    all_histories = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "sparse_active_nodes": [],
        "sparse_total_nodes": [],
        "sparse_relative_cost": [],
        "sparse_score_selected": [],
        "sparse_score_unselected": [],
    }
    per_seed_payload = []
    per_seed_gd = []

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        history, grad_dist_history = run_single_seed(args, seed)

        for key in all_histories:
            all_histories[key].append(history[key])

        per_seed_payload.append(
            {
                "seed": seed,
                "history": history,
            }
        )

        if args.analyze_gradient_dist and grad_dist_history is not None:
            per_seed_gd.append((seed, grad_dist_history))

    result_dir = Path(args.write_results_dir) / exp_name
    save_experiment_results(
        write_results_dir=args.write_results_dir,
        exp_name=exp_name,
        histories=all_histories,
        config=vars(args),
        per_seed_payload=per_seed_payload if args.save_json else None,
    )

    if args.analyze_gradient_dist and per_seed_gd:
        save_gradient_dist_results(result_dir, per_seed_gd, vars(args))

    print(f"\nSaved results to {result_dir}")


if __name__ == "__main__":
    main()