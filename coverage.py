"""Per-layer coverage statistics for sparse ANP/NP training.

Tracks, per training step:
  - mask_coverage        : fraction of nodes directly selected by the mask
  - eff_activity_coverage: fraction of nodes with mean |δa| > threshold
                           (nodes affected by perturbation, including indirect
                           downstream effects when probe_layer is None)
  - mean_layer_norm_sq   : mean over batch of ||δa_l||² for this layer
  - mean_abs_delta_a     : mean |δa| element-wise over batch × nodes

These are computed from the *current* layer.outputs_clean / outputs_noisy
tensors, so they must be called after the noisy forward pass.

Interpretation
--------------
A node's incoming weights receive an ANP update only when its δa is nonzero.
mask_coverage counts nodes *directly* perturbed; eff_activity_coverage counts
nodes *actually affected* (may exceed mask_coverage for layers downstream of a
perturbed layer, and equals it when probe_layer forces a single-layer noise).

Usage in main.py
----------------
    cov = compute_coverage_stats(model, masks, threshold=args.coverage_threshold)
    # cov is a list of dicts, one per layer
    # aggregate into history with update_coverage_history(history, cov)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


COVERAGE_KEYS = (
    "mask_coverage",
    "eff_activity_coverage",
    "mean_layer_norm_sq",
    "mean_abs_delta_a",
)


# ---------------------------------------------------------------------------
# Core stats function
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    model,
    masks: list,
    threshold: float = 1e-12,
) -> list[dict]:
    """Return per-layer coverage stats for the current training step.

    Parameters
    ----------
    model : MLP
        Must have layer.outputs_clean and layer.outputs_noisy populated
        (i.e., the clean and noisy forward passes have already run).
    masks : list[tf.Tensor]
        One [1, units] float32 binary mask per layer.
    threshold : float
        A node counts as 'effectively active' when its mean |δa| across the
        batch exceeds this value.  Default 1e-12 catches any non-zero δa.

    Returns
    -------
    list[dict]  — one dict per layer with keys in COVERAGE_KEYS.
    """
    stats: list[dict] = []
    for layer, mask in zip(model.layers_list, masks):
        mask_np = mask.numpy().reshape(-1)
        mask_cov = float(mask_np.mean())

        if layer.outputs_clean is not None and layer.outputs_noisy is not None:
            delta_a = layer.outputs_noisy - layer.outputs_clean   # [B, H]
            abs_da  = tf.abs(delta_a)

            # Per-node mean |δa| across the batch
            node_mean_abs = tf.reduce_mean(abs_da, axis=0)        # [H]
            eff_cov = float(
                tf.reduce_mean(
                    tf.cast(node_mean_abs > threshold, tf.float32)
                ).numpy()
            )

            # Per-sample layer norm-squared, then mean over batch
            layer_ns_sq = tf.reduce_sum(abs_da ** 2, axis=1)      # [B]
            mean_ns_sq  = float(tf.reduce_mean(layer_ns_sq).numpy())
            mean_abs    = float(tf.reduce_mean(abs_da).numpy())
        else:
            eff_cov    = float("nan")
            mean_ns_sq = float("nan")
            mean_abs   = float("nan")

        stats.append({
            "mask_coverage":         mask_cov,
            "eff_activity_coverage": eff_cov,
            "mean_layer_norm_sq":    mean_ns_sq,
            "mean_abs_delta_a":      mean_abs,
        })
    return stats


# ---------------------------------------------------------------------------
# History helpers (used in main.py)
# ---------------------------------------------------------------------------

def empty_coverage_history(n_layers: int) -> dict:
    """Return a per-epoch accumulator: {layer{l}_{key}: []} for all layers/keys."""
    return {
        f"layer{l}_{k}": []
        for l in range(n_layers)
        for k in COVERAGE_KEYS
    }


def update_coverage_history(
    cov_history: dict,
    epoch_means: list[dict],
) -> None:
    """Append one epoch's mean coverage stats to the running history dict."""
    for l, layer_stats in enumerate(epoch_means):
        for k in COVERAGE_KEYS:
            cov_history[f"layer{l}_{k}"].append(layer_stats.get(k, float("nan")))


# ---------------------------------------------------------------------------
# Epoch aggregation
# ---------------------------------------------------------------------------

def mean_coverage_stats(batch_stats_list: list[list[dict]]) -> list[dict]:
    """Average a list of per-batch coverage-stats lists into one epoch mean.

    Parameters
    ----------
    batch_stats_list : list[list[dict]]
        Outer list = batches; inner list = layers.

    Returns
    -------
    list[dict]  — one dict per layer with epoch-mean values.
    """
    if not batch_stats_list:
        return []
    n_layers = len(batch_stats_list[0])
    result = []
    for l in range(n_layers):
        agg: dict[str, list[float]] = {k: [] for k in COVERAGE_KEYS}
        for batch in batch_stats_list:
            for k in COVERAGE_KEYS:
                v = batch[l].get(k, float("nan"))
                if v == v:   # skip NaN
                    agg[k].append(v)
        result.append(
            {k: (float(np.mean(agg[k])) if agg[k] else float("nan")) for k in COVERAGE_KEYS}
        )
    return result
