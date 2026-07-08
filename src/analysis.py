"""
Gradient distribution analysis for ANP / DANP.

Measures how concentrated or diffuse the per-node learning signal is by
computing distribution statistics of the effective-error magnitude

    g_i = |activity_diff_i * performance_diff / ||activity_diff||²|

across all hidden nodes after each training batch.  A concentrated distribution
(high CV, top-10 % energy fraction > 0.5, high Gini) suggests that adaptive
node selection could outperform random sparse perturbation.  A diffuse
distribution (CV ≈ 0, top-10 % ≈ 0.1, low Gini) means the signal is spread
evenly and random selection is a near-optimal unbiased estimator.

Results are saved per epoch, per seed, and averaged across seeds to
``<exp_dir>/gradient_dist.json``.  Enable with ``--analyze_gradient_dist``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# Keys produced by compute_gradient_dist_stats (and expected by all callers).
GD_STAT_KEYS = ("g_mean", "g_std", "g_cv", "top_1pct", "top_5pct", "top_10pct", "top_25pct", "gini")


def empty_grad_dist_history() -> dict:
    """Return a per-epoch accumulator (list-per-key) for gradient distribution stats."""
    return {k: [] for k in GD_STAT_KEYS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _network_norm_sq(model, eps: float = 1e-12) -> tf.Tensor:
    """Per-sample squared L2 norm of the full activity-diff vector. Shape [B, 1].

    Uses the current layer.outputs_noisy / layer.outputs_clean values.
    Adds eps to avoid division-by-zero in the analysis (does NOT affect training).
    """
    diffs = []
    for layer in model.layers_list:
        diff = layer.outputs_noisy - layer.outputs_clean
        diffs.append(tf.reshape(diff, [tf.shape(diff)[0], -1]))
    act_diff = tf.concat(diffs, axis=1)
    norm_sq = tf.expand_dims(tf.linalg.norm(act_diff, axis=1), axis=-1) ** 2
    return tf.maximum(norm_sq, eps)


def _gini(x: tf.Tensor) -> float:
    """Gini coefficient of a non-negative 1-D tensor.

    Returns 0.0 for perfect equality (all nodes carry equal signal),
    approaching 1.0 for perfect concentration (one node carries everything).

    Formula: G = (2 * Σ_i rank_i * x_i) / (n * Σ x_i) - (n+1)/n
    where values are sorted ascending and rank_i = 1..n.
    """
    x = tf.cast(tf.sort(tf.reshape(x, [-1])), tf.float64)
    total = float(tf.reduce_sum(x).numpy())
    if total == 0.0:
        return float("nan")
    n = tf.cast(tf.shape(x)[0], tf.float64)
    rank = tf.cast(tf.range(1, tf.shape(x)[0] + 1), tf.float64)
    return float(((2.0 * tf.reduce_sum(rank * x)) / (n * total) - (n + 1.0) / n).numpy())


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def compute_gradient_dist_stats(model, performance_diff: tf.Tensor) -> dict:
    """Compute distribution statistics of the per-node ANP effective-error magnitude.

    For each node i in layer l the ANP gradient is proportional to:

        g_i = activity_diff_i * performance_diff / ||activity_diff||²

    This function estimates |g_i| per node by averaging over the batch, then
    computes concentration statistics over all nodes in the network.

    Parameters
    ----------
    model:
        MLP with populated layer.outputs_clean / layer.outputs_noisy (from the
        current clean + noisy forward passes).
    performance_diff:
        Tensor of shape [B, 1] = loss_clean - loss_noisy.
        Pass the RAW value BEFORE any variant-specific scaling (e.g. NP's 1/σ²).

    Returns
    -------
    dict with keys defined in GD_STAT_KEYS, or {} if the pass state is invalid.

    Key                 Interpretation
    -------             ---------------------------
    g_mean              Mean |g_i| across all nodes (scale of update signal)
    g_std               Std of |g_i|  (spread)
    g_cv                Coefficient of variation = g_std / g_mean
                        CV ≈ 0  → nearly uniform; CV ≫ 1 → highly concentrated
    top_1pct ..         Fraction of total energy carried by the top 1/5/10/25% of nodes
    top_25pct           If top_10% > ~0.5 → concentrated; if top_10% ≈ 0.1 → diffuse
    gini                Gini coefficient (0 = equal, ~1 = one node dominates)
    n_nodes             Total number of nodes analysed (sanity check)
    """
    network_norm_sq = _network_norm_sq(model)  # [B, 1]

    node_scores = []
    for layer in model.layers_list:
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            continue
        activity_diff = layer.outputs_noisy - layer.outputs_clean  # [B, units]
        g = tf.abs(activity_diff * performance_diff / network_norm_sq)  # [B, units]
        node_scores.append(tf.reduce_mean(g, axis=0))                  # [units]

    if not node_scores:
        return {}

    all_scores = tf.cast(tf.concat(node_scores, axis=0), tf.float32)  # [total_nodes]

    # Filter non-finite values (can arise from degenerate batches or dead layers)
    all_scores = tf.boolean_mask(all_scores, tf.math.is_finite(all_scores))
    n_nodes = int(tf.size(all_scores).numpy())
    if n_nodes == 0:
        return {}

    g_mean = float(tf.reduce_mean(all_scores).numpy())
    g_std = float(tf.math.reduce_std(all_scores).numpy())
    g_cv = (g_std / g_mean) if g_mean > 0.0 else float("nan")

    total_energy = float(tf.reduce_sum(all_scores).numpy())
    sorted_desc = tf.sort(all_scores, direction="DESCENDING")

    def top_k_frac(pct: float) -> float:
        k = max(1, int(round(n_nodes * pct)))
        top_energy = float(tf.reduce_sum(sorted_desc[:k]).numpy())
        return (top_energy / total_energy) if total_energy > 0.0 else float("nan")

    return {
        "g_mean":    g_mean,
        "g_std":     g_std,
        "g_cv":      g_cv,
        "top_1pct":  top_k_frac(0.01),
        "top_5pct":  top_k_frac(0.05),
        "top_10pct": top_k_frac(0.10),
        "top_25pct": top_k_frac(0.25),
        "gini":      _gini(all_scores),
        "n_nodes":   n_nodes,
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_gradient_dist_results(
    result_dir: Path,
    per_seed_histories: list[tuple[int, dict]],
    config: dict,
) -> None:
    """Save per-seed per-epoch gradient distribution stats + cross-seed summary to JSON.

    per_seed_histories: list of (seed_int, grad_dist_history_dict) pairs.
    """
    def _to_json_val(v):
        return None if (v is None or v != v) else float(v)  # nan → null

    seeds_data = []
    for seed, gd_hist in per_seed_histories:
        n_epochs = len(next(iter(gd_hist.values())))
        epochs_data = []
        for ep in range(n_epochs):
            row = {"epoch": ep}
            for k in GD_STAT_KEYS:
                row[k] = _to_json_val(gd_hist[k][ep])
            epochs_data.append(row)
        seeds_data.append({"seed": seed, "per_epoch": epochs_data})

    # Mean across seeds per epoch (NaN-safe)
    n_epochs = len(per_seed_histories[0][1][GD_STAT_KEYS[0]])
    mean_across: dict = {}
    for k in GD_STAT_KEYS:
        matrix = np.array(
            [[gd[k][ep] for ep in range(n_epochs)] for _, gd in per_seed_histories],
            dtype=float,
        )
        mean_across[k] = [_to_json_val(v) for v in np.nanmean(matrix, axis=0).tolist()]

    payload = {
        "config": config,
        "interpretation": {
            "g_cv":      "coefficient of variation; ~0 = uniform signal, >>1 = concentrated",
            "top_10pct": ">0.5 = concentrated (top 10% of nodes carry majority); ~0.1 = diffuse",
            "gini":      "0 = perfectly equal signal; ~1 = one node dominates",
        },
        "mean_across_seeds": mean_across,
        "seeds": seeds_data,
    }

    out_path = result_dir / "gradient_dist.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Gradient distribution stats saved to {out_path}")
