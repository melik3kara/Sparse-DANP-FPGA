"""
Equivalence test: linearized_sparse_aware vs linearized

For the same model weights, batch, random seed, and sparse mask, the two paths
must produce identical delta_z tensors (and therefore identical weight updates).

Run:
    python test_sal_equivalence.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import tensorflow as tf

from models import MLP
from sparse import compute_sparse_masks
from algorithms import linearized_activity_diffs, linearized_activity_diffs_sparse_aware

TOLERANCE_DELTA_Z  = 1e-5   # per-element, all layers
TOLERANCE_UPDATES  = 1e-5   # per-element, all weight tensors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_clean_forward(model, x):
    model.forward(x, decorrelate=False)


def _apply_noise(model, masks, noise_std=0.1, seed=42):
    tf.random.set_seed(seed)
    model.reset_all_noise(noise_std, masks=masks, noise_distribution="gaussian")


def _np_weight_updates(model, performance_diff):
    """
    Compute ΔW = (1/B) Σ_b performance_diff_b * delta_a_prev_b^T  @  delta_a_b
    This is the NP-style update: grads = δa_l^T @ error (outer-product rule).

    For a fair comparison we use the cached outputs_noisy that are set to
    outputs_clean + delta_z_l by the sparse-aware path (same values in both).
    """
    updates = []
    n_layers = len(model.layers_list)
    for i, layer in enumerate(model.layers_list):
        delta_a_l = layer.outputs_noisy - layer.activation_fn(layer.outputs_clean)
        if i == 0:
            x_in = layer.inputs_clean
        else:
            prev = model.layers_list[i - 1]
            x_in = prev.activation_fn(prev.outputs_noisy)
        dW = tf.reduce_mean(
            tf.expand_dims(performance_diff, 2) *
            tf.einsum("bi,bj->bij", x_in, delta_a_l),
            axis=0,
        )
        updates.append(dW.numpy())
    return updates


# ---------------------------------------------------------------------------
# Single equivalence check
# ---------------------------------------------------------------------------

def check_equivalence(
    hidden_sizes: list[int],
    batch_size: int,
    input_dim: int,
    output_dim: int,
    sparse_fraction: float,
    selected_layer_idx: int,
    noise_std: float = 0.1,
    seed: int = 0,
    label: str = "",
) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = MLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
    )
    n_layers = len(model.layers_list)

    x = tf.random.normal([batch_size, input_dim], seed=seed)
    y = tf.one_hot(
        tf.random.uniform([batch_size], minval=0, maxval=output_dim,
                          dtype=tf.int32, seed=seed),
        output_dim,
    )

    _run_clean_forward(model, x)

    # Build scheduled_layer masks: only selected_layer_idx gets non-zero masks.
    # scheduled_layer uses step % n_layers as the selected layer.
    # We pick step = selected_layer_idx so the desired layer is selected.
    step = selected_layer_idx
    masks = compute_sparse_masks(
        model=model,
        fraction=sparse_fraction,
        policy="scheduled_layer",
        step=step,
    )

    tf.random.set_seed(seed + 1)
    model.reset_all_noise(noise_std, masks=masks, noise_distribution="gaussian")

    # --- Dense path ---
    _run_clean_forward(model, x)
    delta_z_dense, y_dense = linearized_activity_diffs(model)

    # Capture model outputs after dense path for update computation.
    for layer, dz in zip(model.layers_list, delta_z_dense):
        layer.outputs_noisy = layer.outputs_clean + dz

    perf_diff = tf.reduce_mean(y_dense - y, axis=1, keepdims=True)  # placeholder signal
    updates_dense = _np_weight_updates(model, perf_diff)

    # --- Sparse-aware path (same noise already set) ---
    _run_clean_forward(model, x)
    delta_z_sparse, y_sparse, cost_stats = linearized_activity_diffs_sparse_aware(model)

    for layer, dz in zip(model.layers_list, delta_z_sparse):
        layer.outputs_noisy = layer.outputs_clean + dz

    perf_diff2 = tf.reduce_mean(y_sparse - y, axis=1, keepdims=True)
    updates_sparse = _np_weight_updates(model, perf_diff2)

    # --- Compare delta_z ---
    max_delta_z_diff = 0.0
    for i, (dz_d, dz_s) in enumerate(zip(delta_z_dense, delta_z_sparse)):
        diff = float(tf.reduce_max(tf.abs(dz_d - dz_s)).numpy())
        max_delta_z_diff = max(max_delta_z_diff, diff)

    # --- Compare y_noisy_approx ---
    max_y_diff = float(tf.reduce_max(tf.abs(y_dense - y_sparse)).numpy())

    # --- Compare weight updates ---
    max_update_diff = 0.0
    for ud, us in zip(updates_dense, updates_sparse):
        diff = float(np.max(np.abs(ud - us)))
        max_update_diff = max(max_update_diff, diff)

    passed = (max_delta_z_diff < TOLERANCE_DELTA_Z and
              max_update_diff  < TOLERANCE_UPDATES)

    return {
        "label":            label,
        "selected_layer":   selected_layer_idx,
        "sparse_fraction":  sparse_fraction,
        "max_delta_z_diff": max_delta_z_diff,
        "max_y_diff":       max_y_diff,
        "max_update_diff":  max_update_diff,
        "pass":             passed,
        "mac_saving_ratio": cost_stats["sal_mac_saving_ratio"],
        "skipped_prefix":   cost_stats["sal_skipped_prefix_layers"],
        "active_nodes":     cost_stats["sal_active_nodes_selected"],
        "dense_macs":       cost_stats["sal_dense_macs"],
        "sparse_macs":      cost_stats["sal_sparse_aware_macs"],
    }


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def main():
    INPUT_DIM   = 784
    OUTPUT_DIM  = 10
    HIDDEN      = [128, 64]
    BATCH       = 32

    n_layers = len(HIDDEN) + 1  # hidden + output

    cases = []
    for frac in [0.25, 0.5, 0.75, 1.0]:
        for sel in range(n_layers):
            for seed in [0, 1, 2]:
                cases.append(dict(
                    hidden_sizes=HIDDEN,
                    batch_size=BATCH,
                    input_dim=INPUT_DIM,
                    output_dim=OUTPUT_DIM,
                    sparse_fraction=frac,
                    selected_layer_idx=sel,
                    noise_std=0.1,
                    seed=seed,
                    label=f"frac={frac} sel_layer={sel} seed={seed}",
                ))

    print(f"Running {len(cases)} equivalence checks...")
    print(f"  Architecture: {INPUT_DIM} → {HIDDEN} → {OUTPUT_DIM}")
    print(f"  Batch size: {BATCH}")
    print(f"  Tolerance: delta_z < {TOLERANCE_DELTA_Z}, updates < {TOLERANCE_UPDATES}")
    print()

    results = []
    for c in cases:
        r = check_equivalence(**c)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"  [{status}] {r['label']:<40}  "
            f"max_δz={r['max_delta_z_diff']:.2e}  "
            f"max_ΔW={r['max_update_diff']:.2e}  "
            f"mac_saving={r['mac_saving_ratio']:.3f}  "
            f"skip={int(r['skipped_prefix'])}"
        )

    n_pass = sum(r["pass"] for r in results)
    n_fail = len(results) - n_pass
    print()
    print(f"Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    print()

    if n_fail > 0:
        print("FAILED cases:")
        for r in results:
            if not r["pass"]:
                print(f"  {r['label']}  delta_z={r['max_delta_z_diff']:.2e}  "
                      f"update={r['max_update_diff']:.2e}")

    # Summary table: aggregate over seeds for a clean report
    print()
    print("MAC saving summary (max over seeds):")
    print(f"  {'sparse_frac':<12} {'sel_layer':<12} {'mac_saving':<12} {'skipped':<10} "
          f"{'dense_macs':<12} {'sparse_macs':<12}")
    seen = set()
    for r in results:
        key = (r["sparse_fraction"], r["selected_layer"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r['sparse_fraction']:<12.2f} {int(r['selected_layer']):<12} "
              f"{r['mac_saving_ratio']:<12.4f} {int(r['skipped_prefix']):<10} "
              f"{int(r['dense_macs']):<12} {int(r['sparse_macs']):<12}")

    overall_max_dz  = max(r["max_delta_z_diff"] for r in results)
    overall_max_dw  = max(r["max_update_diff"] for r in results)
    print()
    print(f"Overall max_abs_delta_diff = {overall_max_dz:.2e}  "
          f"(tolerance {TOLERANCE_DELTA_Z:.0e})")
    print(f"Overall max_abs_update_diff = {overall_max_dw:.2e}  "
          f"(tolerance {TOLERANCE_UPDATES:.0e})")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
