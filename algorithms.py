from __future__ import annotations

import tensorflow as tf

from sparse import (
    compute_sparse_masks,
    compute_activity_diff_topk_masks,
    compute_activity_diff_score_stats,
    compute_activity_loss_topk_masks,
    compute_activity_loss_score_stats,
    compute_gradient_aligned_topk_masks,
    compute_gradient_aligned_score_stats,
    mask_stats,
    build_probe_masks,
)
from analysis import compute_gradient_dist_stats
from coverage import compute_coverage_stats

COMPARE_DIAG_KEYS = (
    "rel_delta_z_error",
    "rel_delta_logits_error",
    "delta_L_noisy",
    "delta_L_linearized",
    "abs_delta_L_error",
    "rel_delta_L_error",
)

# Analytical cost-accounting keys for delta_mode=linearized_sparse_aware.
SAL_COST_KEYS = (
    "sal_dense_macs",
    "sal_sparse_aware_macs",
    "sal_mac_saving_ratio",
    "sal_selected_layer",
    "sal_skipped_prefix_layers",
    "sal_active_nodes_selected",
    "sal_active_node_ratio_selected",
    "sal_dense_matmuls",
    "sal_sparse_matvecs",
)


def apply_decorrelation_update(model, decor_lr: float) -> None:
    """
    Per-layer decorrelation update:
        C = (X^T X / B) ⊙ (1 - I)
        R <- R - eta_dec * C R
    """
    for layer in model.layers_list:
        x = layer.inputs_clean
        batch_size = tf.cast(tf.shape(x)[0], x.dtype)
        eye = tf.eye(tf.shape(x)[1], dtype=x.dtype)
        corr = tf.einsum("ni,nj->ij", x, x) / batch_size
        corr = corr * (1.0 - eye)
        update = -tf.einsum("ij,jk->ik", corr, layer.R)
        layer.R.assign_add(tf.cast(decor_lr, layer.R.dtype) * tf.cast(update, layer.R.dtype))


def bp_gradients(model, x, y, loss_fn, decorrelated: bool):
    with tf.GradientTape() as tape:
        y_pred = model.forward(x, decorrelate=decorrelated)
        loss_per_sample = loss_fn(y_pred, y)
        loss = tf.reduce_mean(loss_per_sample)
    grads = tape.gradient(loss, model.ordered_trainable_variables())
    return grads, y_pred, loss_per_sample


def _dense_weight_and_bias_grad(layer, error: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = layer.inputs_clean
    batch_size = tf.cast(tf.shape(x)[0], x.dtype)
    w_grad = -(1.0 / batch_size) * tf.einsum("ni,nj->ij", x, error)
    b_grad = -(1.0 / batch_size) * tf.reduce_sum(error, axis=0)
    return w_grad, b_grad


def _network_activity_stats(model, masks: list[tf.Tensor] | None = None):
    """
    masks: optional list of per-layer masks with shape [1, units], one per
    layer in model.layers_list. When provided, activity differences for
    unselected nodes are zeroed out before computing the norm, and n_total
    counts only the selected ("active") nodes across the whole network.
    """
    diffs = []
    for i, layer in enumerate(model.layers_list):
        diff = layer.outputs_noisy - layer.outputs_clean
        if masks is not None:
            diff = diff * tf.cast(masks[i], diff.dtype)
        diffs.append(tf.reshape(diff, [tf.shape(diff)[0], -1]))
    act_diff = tf.concat(diffs, axis=1)
    norm_sq = tf.expand_dims(tf.linalg.norm(act_diff, axis=1), axis=-1) ** 2
    if masks is not None:
        n_total = tf.add_n([tf.reduce_sum(tf.cast(m, tf.float32)) for m in masks])
    else:
        n_total = tf.cast(tf.shape(act_diff)[1], tf.float32)
    return norm_sq, n_total


def _count_active_nodes(model, masks: list | None) -> tf.Tensor:
    """Number of active nodes across the network — cheap, no norm computation."""
    if masks is not None:
        return tf.add_n([tf.reduce_sum(tf.cast(m, tf.float32)) for m in masks])
    return tf.cast(sum(layer.units for layer in model.layers_list), tf.float32)


def _np_like_grads_from_cached_pass(
    model,
    performance_diff: tf.Tensor,
    variant: str,
    layer_idx: int | None = None,
    masks: list[tf.Tensor] | None = None,
    norm_mode: str = "exact",
    norm_sq_override: tf.Tensor | None = None,
    norm_eps: float = 1e-8,
) -> list[tf.Tensor]:
    """
    variant in {"np", "anp", "inp"}.
    Returns grads aligned with model.ordered_trainable_variables().

    masks: optional list of per-layer masks with shape [1, units] used by the
    sparse NP/ANP/DANP variants. For "np", layer.noise is already zeroed for
    unselected nodes (see model.reset_all_noise), so error is automatically
    sparse. For "anp", activity_diff is masked here and the normalization in
    _network_activity_stats is restricted to the selected nodes.

    norm_mode controls the ‖δa‖² normalizer in the ANP update:
      'exact'  — per-step per-sample ‖δa‖² (current default; requires a
                 global reduction every step + per-sample division).
      'ema'    — scalar EMA of ‖δa‖² passed as norm_sq_override; global
                 reduction amortised by norm_update_every, per-step cost is
                 one scalar multiply (stored reciprocal in hardware).
      'none'   — no normalizer; scale = δL (diagnostic only; changes LR/noise
                 sensitivity and removes both reduction and division).
    """
    if variant == "anp" and norm_mode != "none":
        if norm_sq_override is None:
            # exact: per-step per-sample norm from current outputs_noisy
            network_norm_sq, n_total = _network_activity_stats(model, masks=masks)
        else:
            # ema: norm provided as scalar; still need n_total for scale compensation
            n_total = _count_active_nodes(model, masks)
            network_norm_sq = norm_sq_override  # scalar broadcasts over [batch, units]
    elif variant == "np":
        # np uses noise * performance_diff; network_norm_sq not needed
        network_norm_sq, n_total = None, None

    grads = []
    for i, layer in enumerate(model.layers_list):
        compute_this_layer = (layer_idx is None) or (layer_idx == i)

        if not compute_this_layer:
            kernel, bias = layer.trainable_variables
            grads.extend([tf.zeros_like(kernel), tf.zeros_like(bias)])
            continue

        if variant == "np":
            error = layer.noise * performance_diff

        elif variant == "anp":
            activity_diff = layer.outputs_noisy - layer.outputs_clean
            if masks is not None:
                activity_diff = activity_diff * tf.cast(masks[i], activity_diff.dtype)
            error = activity_diff * performance_diff
            if norm_mode != "none":
                # exact: no eps (preserves original behaviour); ema: add eps for safety
                denom = network_norm_sq + (
                    tf.cast(0.0, error.dtype) if norm_mode == "exact"
                    else tf.cast(norm_eps, error.dtype)
                )
                error = error / denom
                error = error * tf.cast(n_total, error.dtype)
            # norm_mode='none': use error = δa × δL with no scale compensation

        elif variant == "inp":
            activity_diff = layer.outputs_noisy - layer.outputs_clean
            layer_norm_sq = tf.expand_dims(tf.linalg.norm(activity_diff, axis=1), axis=-1) ** 2
            n_layer = tf.cast(tf.shape(activity_diff)[1], activity_diff.dtype)
            error = activity_diff * performance_diff
            error = error / layer_norm_sq
            error = error * n_layer

        else:
            raise ValueError(f"Unknown variant: {variant}")

        w_grad, b_grad = _dense_weight_and_bias_grad(layer, error)
        kernel, bias = layer.trainable_variables
        grads.extend([tf.cast(w_grad, kernel.dtype), tf.cast(b_grad, bias.dtype)])

    return grads


def _update_ema_norm( #exponential moving average of network norm for ANP normalization
    model,
    masks,
    norm_state: dict,
    step: int,
    norm_beta: float,
    norm_update_every: int,
) -> tf.Tensor:
    """
    Update the EMA of ‖δa‖² and return the current EMA as a scalar tf.float32.

    Requires outputs_noisy to be set (call after a forward_noisy pass).
    Only recomputes ‖δa‖² every norm_update_every steps — the other steps reuse
    the stored EMA, amortising the global reduction cost by norm_update_every.
    """
    if step % norm_update_every == 0:
        norm_sq, _ = _network_activity_stats(model, masks=masks)
        current_val = float(tf.reduce_mean(norm_sq).numpy())
        if norm_state["ema_norm"] is None:
            norm_state["ema_norm"] = current_val
        else:
            norm_state["ema_norm"] = (
                norm_beta * norm_state["ema_norm"] +
                (1.0 - norm_beta) * current_val
            )
    ema_val = norm_state["ema_norm"] if norm_state["ema_norm"] is not None else 1.0
    return tf.constant(float(ema_val), dtype=tf.float32)


# ---------------------------------------------------------------------------
# Linearized ANP helpers
# ---------------------------------------------------------------------------

def _activation_derivative_elementwise(activation_fn, z: tf.Tensor) -> tf.Tensor:
    """
    Elementwise activation derivative df/dz via automatic differentiation.

    Works for leaky_relu, relu, tanh, sigmoid (any elementwise activation).
    Do NOT call for softmax — the output layer is handled separately in
    linearized_activity_diffs by applying the true activation to the approximate
    noisy pre-activation rather than linearising around it.
    """
    with tf.GradientTape() as tape:
        tape.watch(z)
        a = activation_fn(z)
    fprime = tape.gradient(a, z)
    if fprime is None:
        raise NotImplementedError(
            f"Cannot compute an elementwise derivative for {activation_fn}. "
            "Only elementwise activations (leaky_relu, relu, tanh, sigmoid, linear) "
            "are supported for hidden layers in linearized mode."
        )
    return fprime


def linearized_activity_diffs(model):
    """
    First-order linearized approximation of pre-activation differences δz_l.

    Requires a completed clean forward pass (sets layer.outputs_clean,
    layer.inputs_clean) and noise set via reset_all_noise (sets layer.noise).

    Propagation (matches clean-pass orientation z = x @ W + b):
        delta_a_0 = 0
        for each layer l:
            delta_z_l = delta_a_{l-1} @ W_l + epsilon_l    # [batch, units_l]
            delta_a_l = f'(z_l) * delta_z_l                # hidden layers only

    Masked nodes already have layer.noise = 0 (set by reset_noise), so no
    additional masking is needed here.

    Returns:
        delta_z_list   — list[tf.Tensor] shape [batch, units_l], one per layer;
                         these are the linearized analogues of
                         (layer.outputs_noisy - layer.outputs_clean).
        y_noisy_approx — [batch, n_classes] approx noisy output: applies the
                         true output activation to (z_L + delta_z_L) rather
                         than linearising the output activation itself.
    """
    n_layers = len(model.layers_list)
    delta_a_prev = None   # None = zero; avoids a wasted matmul at layer 0
    delta_z_list = []

    for i, layer in enumerate(model.layers_list):
        epsilon_l = layer.noise          # [batch, units_l]; already sparse-masked
        is_last = (i == n_layers - 1)

        if delta_a_prev is None:
            delta_z_l = epsilon_l        # first layer: no propagation from prev
        else:
            delta_z_l = tf.matmul(delta_a_prev, layer.kernel) + epsilon_l

        delta_z_list.append(delta_z_l)

        if not is_last:
            z_l = layer.outputs_clean    # clean pre-activation cached by clean forward
            fprime_l = _activation_derivative_elementwise(layer.activation_fn, z_l)
            delta_a_prev = fprime_l * delta_z_l

    last = model.layers_list[-1]
    y_noisy_approx = last.activation_fn(last.outputs_clean + delta_z_list[-1])
    return delta_z_list, y_noisy_approx


def linearized_activity_diffs_sparse_aware(model):
    """
    Sparse-aware variant of linearized_activity_diffs.

    Produces the same delta_z_list as linearized_activity_diffs for the same
    model state (same epsilon tensors), but skips multiply-accumulate operations
    that are guaranteed to be zero by the sparse mask structure:

    1. Prefix skipping: layers 0..s-1 whose noise is all-zero contribute
       delta_a_l = 0.  Their propagation matmuls (0 @ W) are skipped entirely.
    2. Zero-input injection: at the injection layer s, delta_z_s = epsilon_s
       (no matmul, since delta_a_{s-1} = 0).
    3. Sparse first-propagation: the first matmul after injection uses only
       the k_s active-node columns of delta_a_s, saving (H_s - k_s) * H_{s+1}
       MACs per batch sample.
    4. Remaining layers: standard dense propagation (delta is no longer sparse).

    Correctness: the dense path also computes delta_a_l = f'(z_l) * 0 = 0 for
    prefix layers (since epsilon_l = 0), and delta_z_s = 0 @ W_s + epsilon_s =
    epsilon_s, so the results are numerically identical.  The only potential
    floating-point discrepancy is in the sparse matvec vs the equivalent dense
    matmul of a zero-sparse matrix; in IEEE 754 arithmetic these are both exact.

    Returns
    -------
    delta_z_list : list[tf.Tensor] — same shapes as linearized_activity_diffs.
    y_noisy_approx : tf.Tensor
    cost_stats : dict — keys match SAL_COST_KEYS; all counts are analytical,
                        not measured wall-clock time.
    """
    n_layers = len(model.layers_list)
    batch_size = int(tf.shape(model.layers_list[0].inputs_clean)[0].numpy())

    # --- Analytical MAC count for the dense linearized baseline ---
    # Layer 0 is already free in dense mode (delta_a_prev=None shortcut), so
    # dense MACs start from layer 1.
    dense_macs = sum(
        batch_size * model.layers_list[l - 1].units * model.layers_list[l].units
        for l in range(1, n_layers)
    )

    # --- Find first perturbed layer (injection point s) ---
    first_active = None
    for i, layer in enumerate(model.layers_list):
        if float(tf.reduce_sum(tf.abs(layer.noise)).numpy()) > 0.0:
            first_active = i
            break

    if first_active is None:
        # No perturbation at all — all deltas zero, zero MACs.
        delta_z_list = [
            tf.zeros([batch_size, layer.units], dtype=tf.float32)
            for layer in model.layers_list
        ]
        last = model.layers_list[-1]
        y_noisy_approx = last.activation_fn(last.outputs_clean)
        cost_stats = {
            "sal_dense_macs":               dense_macs,
            "sal_sparse_aware_macs":        0,
            "sal_mac_saving_ratio":         1.0,
            "sal_selected_layer":           -1,
            "sal_skipped_prefix_layers":    n_layers,
            "sal_active_nodes_selected":    0,
            "sal_active_node_ratio_selected": 0.0,
            "sal_dense_matmuls":            n_layers - 1,
            "sal_sparse_matvecs":           0,
        }
        return delta_z_list, y_noisy_approx, cost_stats

    s = first_active
    delta_z_list: list = [None] * n_layers
    sparse_aware_macs = 0
    dense_matmuls_executed = 0
    sparse_matvecs_executed = 0

    # Prefix layers: zero delta_z (no computation).
    for i in range(s):
        delta_z_list[i] = tf.zeros(
            [batch_size, model.layers_list[i].units],
            dtype=model.layers_list[i].noise.dtype,
        )

    # Injection layer s: delta_z_s = epsilon_s (no matmul; delta_a_prev = 0).
    layer_s = model.layers_list[s]
    delta_z_s = layer_s.noise
    delta_z_list[s] = delta_z_s

    # Number of active nodes in injection layer (for sparse first-propagation).
    mask_s = layer_s.noise_mask  # [1, H_s] or None
    if mask_s is not None:
        k_s = int(tf.reduce_sum(tf.cast(mask_s > 0, tf.int32)).numpy())
    else:
        k_s = layer_s.units

    if s < n_layers - 1:
        # delta_a_s is sparse: non-zero only at the k_s active positions.
        z_s = layer_s.outputs_clean
        fprime_s = _activation_derivative_elementwise(layer_s.activation_fn, z_s)
        delta_a_s = fprime_s * delta_z_s  # [batch, H_s]

        # First propagation after injection: sparse matvec if k_s < H_s.
        layer_next = model.layers_list[s + 1]
        epsilon_next = layer_next.noise

        if mask_s is not None and k_s < layer_s.units:
            active_idx = tf.cast(
                tf.squeeze(tf.where(tf.squeeze(mask_s, axis=0) > 0), axis=1), tf.int32
            )
            delta_a_s_cols = tf.gather(delta_a_s, active_idx, axis=1)      # [B, k_s]
            W_next_rows = tf.gather(layer_next.kernel, active_idx, axis=0)  # [k_s, H_{s+1}]
            delta_z_next = tf.matmul(delta_a_s_cols, W_next_rows) + epsilon_next
            sparse_aware_macs += batch_size * k_s * layer_next.units
            sparse_matvecs_executed += 1
        else:
            # k_s == H_s: no column saving, fall back to dense.
            delta_z_next = tf.matmul(delta_a_s, layer_next.kernel) + epsilon_next
            sparse_aware_macs += batch_size * layer_s.units * layer_next.units
            dense_matmuls_executed += 1

        delta_z_list[s + 1] = delta_z_next

        # Remaining layers s+2..L-1: dense propagation (delta is no longer sparse).
        if s + 1 < n_layers - 1:
            z_next = layer_next.outputs_clean
            fprime_next = _activation_derivative_elementwise(
                layer_next.activation_fn, z_next
            )
            delta_a_prev = fprime_next * delta_z_next
        else:
            delta_a_prev = None

        for i in range(s + 2, n_layers):
            layer = model.layers_list[i]
            delta_z_l = tf.matmul(delta_a_prev, layer.kernel) + layer.noise
            delta_z_list[i] = delta_z_l
            sparse_aware_macs += batch_size * model.layers_list[i - 1].units * layer.units
            dense_matmuls_executed += 1
            if i < n_layers - 1:
                z_l = layer.outputs_clean
                fprime_l = _activation_derivative_elementwise(layer.activation_fn, z_l)
                delta_a_prev = fprime_l * delta_z_l

    last = model.layers_list[-1]
    y_noisy_approx = last.activation_fn(last.outputs_clean + delta_z_list[-1])

    mac_saving = (
        (dense_macs - sparse_aware_macs) / dense_macs if dense_macs > 0 else 0.0
    )
    cost_stats = {
        "sal_dense_macs":               dense_macs,
        "sal_sparse_aware_macs":        sparse_aware_macs,
        "sal_mac_saving_ratio":         mac_saving,
        "sal_selected_layer":           s,
        "sal_skipped_prefix_layers":    s,  # layers before s have zero matmul cost
        "sal_active_nodes_selected":    k_s,
        "sal_active_node_ratio_selected": k_s / layer_s.units,
        "sal_dense_matmuls":            dense_matmuls_executed,
        "sal_sparse_matvecs":           sparse_matvecs_executed,
    }
    return delta_z_list, y_noisy_approx, cost_stats


def _compute_compare_diagnostics(
    model,
    delta_z_list: list,
    y_clean: tf.Tensor,
    y_noisy: tf.Tensor,
    y_noisy_approx: tf.Tensor,
    performance_diff_noisy: tf.Tensor,
    performance_diff_lin: tf.Tensor,
    eps: float = 1e-8,
) -> dict:
    """
    Quality metrics comparing the full noisy path to the linearized path.

    Call AFTER model.forward_noisy() and linearized_activity_diffs() with the
    SAME epsilon tensors.  layer.outputs_noisy must still hold the true noisy
    pre-activations (not the patched linearized values) at the time of this call.
    """
    # 1. Relative pre-activation difference error (what ANP actually uses)
    noisy_dz = tf.concat([
        tf.reshape(layer.outputs_noisy - layer.outputs_clean,
                   [tf.shape(layer.outputs_noisy)[0], -1])
        for layer in model.layers_list
    ], axis=1)
    lin_dz = tf.concat([
        tf.reshape(dz, [tf.shape(dz)[0], -1]) for dz in delta_z_list
    ], axis=1)
    err_dz = noisy_dz - lin_dz
    rel_delta_z_error = float(tf.reduce_mean(
        tf.linalg.norm(err_dz, axis=1) /
        (tf.linalg.norm(noisy_dz, axis=1) + eps)
    ).numpy())

    # 2. Relative output-logit difference error
    delta_logits_noisy = y_noisy - y_clean
    delta_logits_lin   = y_noisy_approx - y_clean
    err_logits = delta_logits_noisy - delta_logits_lin
    rel_delta_logits_error = float(tf.reduce_mean(
        tf.linalg.norm(err_logits, axis=1) /
        (tf.linalg.norm(delta_logits_noisy, axis=1) + eps)
    ).numpy())

    # 3. performance_diff comparison (batch mean)
    dL_noisy = float(tf.reduce_mean(performance_diff_noisy).numpy())
    dL_lin   = float(tf.reduce_mean(performance_diff_lin).numpy())
    abs_dL_err = abs(dL_noisy - dL_lin)
    rel_dL_err = abs_dL_err / (abs(dL_noisy) + eps)

    return {
        "rel_delta_z_error":      rel_delta_z_error,
        "rel_delta_logits_error": rel_delta_logits_error,
        "delta_L_noisy":          dL_noisy,
        "delta_L_linearized":     dL_lin,
        "abs_delta_L_error":      abs_dL_err,
        "rel_delta_L_error":      rel_dL_err,
    }


def perturbation_gradients(
    model,
    x,
    y,
    loss_fn,
    decorrelated: bool,
    variant: str,
    noise_std: float,
    num_noise_iters: int = 1,
    sparse: bool = False,
    sparse_fraction: float = 1.0,
    sparse_policy: str = "random",
    step: int = 0,
    analyze: bool = False,
    # --- new parameters ---
    layer_fractions=None,           # list[float] | None; per-layer fractions from allocate_fractions
    probe_layer: int | None = None, # int: perturb only this layer; None: all layers
    log_coverage: bool = False,
    coverage_threshold: float = 1e-12,
    noise_distribution: str = "gaussian",  # "gaussian" or "rademacher"
    noise_sampling: str = "single",        # "single" | "antithetic" | "resample"
    # --- normalizer ablation ---
    norm_mode: str = "exact",             # "exact" | "ema" | "none"
    norm_beta: float = 0.99,
    norm_update_every: int = 1,
    norm_eps: float = 1e-8,
    norm_state: dict | None = None,       # mutable dict {"ema_norm": float|None}
    # --- linearized ANP ---
    delta_mode: str = "noisy",            # "noisy" | "linearized" | "compare"
):
    """
    variant in {"np", "anp", "inp"}.
    Averages gradients across multiple noisy iterations.

    sparse:
        When True (only for variant in {"np","anp"}), only a fraction of nodes
        per layer are perturbed AND contribute to the update.
    layer_fractions:
        Optional list[float] of per-layer active fractions (one per layer in
        model.layers_list), produced by sparse.allocate_fractions().  When set,
        overrides ``sparse_fraction`` for the simple one-pass policies (random,
        scheduled, activation_threshold).  Two-pass adaptive policies still use
        ``sparse_fraction`` uniformly.
    probe_layer:
        When not None, only layer ``probe_layer`` is perturbed and receives a
        weight update; all other layers get zero noise and zero gradient.
        Overrides sparse/layer_fractions for that iteration.
    log_coverage:
        When True, compute per-layer coverage stats after the noisy forward pass
        and return them as the last element of the return tuple.
    coverage_threshold:
        Threshold for effective-activity coverage (see coverage.py).

    Returns
    -------
    (mean_grads, y_clean, loss_clean, sparse_info, grad_dist, coverage_stats, compare_diag)
    coverage_stats is None when log_coverage=False.
    compare_diag is None unless delta_mode='compare' and it is the last noise iter.
    """
    # Clean pass first
    y_clean = model.forward(x, decorrelate=decorrelated)
    loss_clean = loss_fn(y_clean, y)

    all_iter_grads = []
    sparse_info = None
    grad_dist = None
    coverage_stats = None
    compare_diag = None
    sal_cost = None
    use_sparse = sparse and variant in {"np", "anp"}

    # Precompute fair-budget fraction for probe mode.
    # The probed layer receives at most total_budget = round(sparse_fraction * Σ H_l)
    # active nodes.  If the layer is smaller than total_budget all of its nodes are
    # perturbed and probe_budget_capped is set.
    probe_fraction_for_layer = None
    probe_budget_capped = False
    if probe_layer is not None and variant in {"np", "anp"}:
        total_nodes = sum(layer.units for layer in model.layers_list)
        total_budget = max(1, round(sparse_fraction * total_nodes))
        H_probe = model.layers_list[probe_layer].units
        probe_k = min(total_budget, H_probe)
        probe_fraction_for_layer = probe_k / H_probe
        probe_budget_capped = probe_k < total_budget
        if probe_budget_capped and step == 0:
            print(
                f"[probe] Layer {probe_layer} has only {H_probe} nodes, "
                f"less than total_budget={total_budget}. "
                f"All {H_probe} nodes will be perturbed (budget capped to layer size)."
            )

    # Effective fractions: per-layer list takes priority over scalar.
    fractions = layer_fractions if layer_fractions is not None else sparse_fraction

    for it in range(num_noise_iters):
        masks = None
        score_stats = {
            "score_selected_mean": float("nan"),
            "score_unselected_mean": float("nan"),
        }

        if probe_layer is not None and variant in {"np", "anp"}:
            # Build masks fresh each iteration so scheduled policy advances per step.
            masks = build_probe_masks(
                model=model,
                probe_layer=probe_layer,
                probe_fraction=probe_fraction_for_layer,
                policy=sparse_policy,
                step=step * num_noise_iters + it,
            )
            model.reset_all_noise(noise_std, masks=masks, noise_distribution=noise_distribution)

        elif use_sparse:
            if sparse_policy == "activity_diff_topk":
                # Two-pass: provisional full-noise forward → top-k by mean|Δact|
                model.reset_all_noise(noise_std, masks=None, noise_distribution=noise_distribution)
                _ = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                masks = compute_activity_diff_topk_masks(model, fraction=sparse_fraction)
                score_stats = compute_activity_diff_score_stats(model, masks)
                for layer, mask in zip(model.layers_list, masks):
                    layer.noise = layer.noise * tf.cast(mask, layer.noise.dtype)
                    layer.noise_mask = mask

            elif sparse_policy == "activity_loss_topk":
                # Two-pass: provisional forward → top-k by mean|Δact × Δloss|
                model.reset_all_noise(noise_std, masks=None, noise_distribution=noise_distribution)
                y_noisy_prov = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                loss_noisy_prov = loss_fn(y_noisy_prov, y)
                performance_diff_prov = tf.reshape(loss_clean - loss_noisy_prov, [-1, 1])
                masks = compute_activity_loss_topk_masks(
                    model, performance_diff_prov, fraction=sparse_fraction
                )
                score_stats = compute_activity_loss_score_stats(model, masks, performance_diff_prov)
                for layer, mask in zip(model.layers_list, masks):
                    layer.noise = layer.noise * tf.cast(mask, layer.noise.dtype)
                    layer.noise_mask = mask

            elif sparse_policy == "gradient_aligned_topk":
                # Two-pass: provisional forward → top-k by mean|Δact × Δloss / ‖Δact‖²|
                model.reset_all_noise(noise_std, masks=None, noise_distribution=noise_distribution)
                y_noisy_prov = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                loss_noisy_prov = loss_fn(y_noisy_prov, y)
                performance_diff_prov = tf.reshape(loss_clean - loss_noisy_prov, [-1, 1])
                network_norm_sq_prov, _ = _network_activity_stats(model, masks=None)
                masks = compute_gradient_aligned_topk_masks(
                    model, performance_diff_prov, network_norm_sq_prov, fraction=sparse_fraction
                )
                score_stats = compute_gradient_aligned_score_stats(
                    model, masks, performance_diff_prov, network_norm_sq_prov
                )
                for layer, mask in zip(model.layers_list, masks):
                    layer.noise = layer.noise * tf.cast(mask, layer.noise.dtype)
                    layer.noise_mask = mask

            else:
                # One-pass policies (random, scheduled, activation_threshold):
                # support both scalar and per-layer fractions.
                masks = compute_sparse_masks(
                    model,
                    fraction=fractions,
                    policy=sparse_policy,
                    step=step * num_noise_iters + it,
                )
                model.reset_all_noise(noise_std, masks=masks, noise_distribution=noise_distribution)

        else:
            model.reset_all_noise(noise_std, masks=None, noise_distribution=noise_distribution)

        # Build sparse_info from current masks.
        if probe_layer is not None and variant in {"np", "anp"}:
            base_stats = mask_stats(model, masks)
            sparse_info = {
                **base_stats,
                "policy": f"probe_l{probe_layer}",
                "probe_budget_capped": probe_budget_capped,
                **score_stats,
            }
        elif use_sparse and masks is not None:
            base_stats = mask_stats(model, masks)
            sparse_info = {**base_stats, "policy": sparse_policy, **score_stats}

        if variant in {"np", "anp"}:
            if noise_sampling == "single":
                if delta_mode == "noisy":
                    # ── standard single noisy forward pass (default) ──────────────────────
                    y_noisy = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                    loss_noisy = loss_fn(y_noisy, y)
                    performance_diff = tf.reshape(loss_clean - loss_noisy, [-1, 1])

                    if log_coverage and masks is not None and it == num_noise_iters - 1:
                        coverage_stats = compute_coverage_stats(model, masks, coverage_threshold)

                    if analyze and it == num_noise_iters - 1:
                        grad_dist = compute_gradient_dist_stats(model, performance_diff)

                    if variant == "np":
                        performance_diff = performance_diff / tf.cast(noise_std**2, performance_diff.dtype)

                    norm_sq_override_val = None
                    if norm_mode == "ema" and variant == "anp":
                        norm_sq_override_val = _update_ema_norm(
                            model, masks, norm_state, step, norm_beta, norm_update_every
                        )

                    grads = _np_like_grads_from_cached_pass(
                        model=model,
                        performance_diff=performance_diff,
                        variant=variant,
                        layer_idx=probe_layer,
                        masks=masks,
                        norm_mode=norm_mode,
                        norm_sq_override=norm_sq_override_val,
                        norm_eps=norm_eps,
                    )

                elif delta_mode == "linearized":
                    # ── first-order linearized delta propagation; no noisy forward pass ──
                    # Requires: clean forward done, layer.noise set.
                    delta_z_list, y_noisy_approx = linearized_activity_diffs(model)
                    # Patch outputs_noisy so _np_like_grads_from_cached_pass sees
                    # (outputs_noisy - outputs_clean) = delta_z_l — no other changes needed.
                    for layer, dz in zip(model.layers_list, delta_z_list):
                        layer.outputs_noisy = layer.outputs_clean + dz
                    loss_noisy_lin = loss_fn(y_noisy_approx, y)
                    performance_diff = tf.reshape(loss_clean - loss_noisy_lin, [-1, 1])

                    if variant == "np":
                        performance_diff = performance_diff / tf.cast(noise_std**2, performance_diff.dtype)

                    norm_sq_override_val = None
                    if norm_mode == "ema" and variant == "anp":
                        norm_sq_override_val = _update_ema_norm(
                            model, masks, norm_state, step, norm_beta, norm_update_every
                        )

                    grads = _np_like_grads_from_cached_pass(
                        model=model,
                        performance_diff=performance_diff,
                        variant=variant,
                        layer_idx=probe_layer,
                        masks=masks,
                        norm_mode=norm_mode,
                        norm_sq_override=norm_sq_override_val,
                        norm_eps=norm_eps,
                    )

                elif delta_mode == "linearized_sparse_aware":
                    # ── sparse-aware linearized propagation: prefix skip + sparse matvec ──
                    # Identical update to delta_mode=linearized for the same mask; the
                    # difference is only in which matmuls are actually executed.
                    delta_z_list, y_noisy_approx, _sal = linearized_activity_diffs_sparse_aware(model)
                    for layer, dz in zip(model.layers_list, delta_z_list):
                        layer.outputs_noisy = layer.outputs_clean + dz
                    loss_noisy_lin = loss_fn(y_noisy_approx, y)
                    performance_diff = tf.reshape(loss_clean - loss_noisy_lin, [-1, 1])

                    if it == num_noise_iters - 1:
                        sal_cost = _sal

                    if variant == "np":
                        performance_diff = performance_diff / tf.cast(noise_std**2, performance_diff.dtype)

                    norm_sq_override_val = None
                    if norm_mode == "ema" and variant == "anp":
                        norm_sq_override_val = _update_ema_norm(
                            model, masks, norm_state, step, norm_beta, norm_update_every
                        )

                    grads = _np_like_grads_from_cached_pass(
                        model=model,
                        performance_diff=performance_diff,
                        variant=variant,
                        layer_idx=probe_layer,
                        masks=masks,
                        norm_mode=norm_mode,
                        norm_sq_override=norm_sq_override_val,
                        norm_eps=norm_eps,
                    )

                elif delta_mode == "compare":
                    # ── compare mode: train on full noisy path; linearized is diagnostic ──
                    y_noisy = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                    loss_noisy = loss_fn(y_noisy, y)
                    performance_diff = tf.reshape(loss_clean - loss_noisy, [-1, 1])

                    if log_coverage and masks is not None and it == num_noise_iters - 1:
                        coverage_stats = compute_coverage_stats(model, masks, coverage_threshold)

                    if analyze and it == num_noise_iters - 1:
                        grad_dist = compute_gradient_dist_stats(model, performance_diff)

                    # Linearized path uses same layer.noise tensors (already set)
                    delta_z_list, y_noisy_approx = linearized_activity_diffs(model)
                    loss_noisy_lin = loss_fn(y_noisy_approx, y)
                    pd_lin = tf.reshape(loss_clean - loss_noisy_lin, [-1, 1])

                    # Log diagnostics on the last noise iteration only
                    if it == num_noise_iters - 1:
                        compare_diag = _compute_compare_diagnostics(
                            model=model,
                            delta_z_list=delta_z_list,
                            y_clean=y_clean,
                            y_noisy=y_noisy,
                            y_noisy_approx=y_noisy_approx,
                            performance_diff_noisy=performance_diff,
                            performance_diff_lin=pd_lin,
                        )

                    if variant == "np":
                        performance_diff = performance_diff / tf.cast(noise_std**2, performance_diff.dtype)

                    norm_sq_override_val = None
                    if norm_mode == "ema" and variant == "anp":
                        norm_sq_override_val = _update_ema_norm(
                            model, masks, norm_state, step, norm_beta, norm_update_every
                        )

                    grads = _np_like_grads_from_cached_pass(
                        model=model,
                        performance_diff=performance_diff,
                        variant=variant,
                        layer_idx=probe_layer,
                        masks=masks,
                        norm_mode=norm_mode,
                        norm_sq_override=norm_sq_override_val,
                        norm_eps=norm_eps,
                    )

                else:
                    raise ValueError(f"Unknown delta_mode: {delta_mode!r}")

            else:
                # ── two-pass modes: antithetic (+v / −v) or resample (v1, v2)
                # Pass 1: noise already in layer.noise (+v or v1)
                y_noisy_1 = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                loss_noisy_1 = loss_fn(y_noisy_1, y)
                pd_1 = tf.reshape(loss_clean - loss_noisy_1, [-1, 1])

                if log_coverage and masks is not None and it == num_noise_iters - 1:
                    coverage_stats = compute_coverage_stats(model, masks, coverage_threshold)
                if analyze and it == num_noise_iters - 1:
                    grad_dist = compute_gradient_dist_stats(model, pd_1)

                if variant == "np":
                    pd_1 = pd_1 / tf.cast(noise_std**2, pd_1.dtype)

                # EMA norm update from pass 1 activations; reused for both passes
                norm_sq_override_val = None
                if norm_mode == "ema" and variant == "anp":
                    norm_sq_override_val = _update_ema_norm(
                        model, masks, norm_state, step, norm_beta, norm_update_every
                    )

                grads_1 = _np_like_grads_from_cached_pass(
                    model=model, performance_diff=pd_1,
                    variant=variant, layer_idx=probe_layer, masks=masks,
                    norm_mode=norm_mode, norm_sq_override=norm_sq_override_val,
                    norm_eps=norm_eps,
                )

                # Pass 2: prepare noise for second pass
                if noise_sampling == "antithetic":
                    # Flip sign of the SAME noise tensor; mask and selected
                    # nodes are identical — this cancels the even-order term.
                    for layer in model.layers_list:
                        layer.noise = -layer.noise
                else:
                    # resample: fresh independent noise with the SAME mask so
                    # the active-node budget is identical to antithetic.
                    model.reset_all_noise(
                        noise_std, masks=masks,
                        noise_distribution=noise_distribution,
                    )

                y_noisy_2 = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                loss_noisy_2 = loss_fn(y_noisy_2, y)
                pd_2 = tf.reshape(loss_clean - loss_noisy_2, [-1, 1])

                if variant == "np":
                    pd_2 = pd_2 / tf.cast(noise_std**2, pd_2.dtype)

                grads_2 = _np_like_grads_from_cached_pass(
                    model=model, performance_diff=pd_2,
                    variant=variant, layer_idx=probe_layer, masks=masks,
                    norm_mode=norm_mode, norm_sq_override=norm_sq_override_val,
                    norm_eps=norm_eps,
                )

                grads = [0.5 * (g1 + g2) for g1, g2 in zip(grads_1, grads_2)]

        elif variant == "inp":
            grads = [tf.zeros_like(v) for v in model.ordered_trainable_variables()]
            for layer_idx in range(len(model.layers_list)):
                y_noisy = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=layer_idx)
                loss_noisy = loss_fn(y_noisy, y)
                performance_diff = tf.reshape(loss_clean - loss_noisy, [-1, 1])
                layer_grads = _np_like_grads_from_cached_pass(
                    model=model,
                    performance_diff=performance_diff,
                    variant="inp",
                    layer_idx=layer_idx,
                )
                grads = [g0 + g1 for g0, g1 in zip(grads, layer_grads)]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        all_iter_grads.append(grads)

    mean_grads = [
        tf.reduce_mean(tf.stack([glist[i] for glist in all_iter_grads], axis=0), axis=0)
        for i in range(len(all_iter_grads[0]))
    ]
    return mean_grads, y_clean, loss_clean, sparse_info, grad_dist, coverage_stats, compare_diag, sal_cost


def optimizer_from_name(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def train_step(
    model,
    optimizer,
    x,
    y,
    loss_fn,
    algorithm: str,
    decorrelated: bool,
    noise_std: float,
    num_noise_iters: int,
    decor_lr: float,
    sparse: bool = False,
    sparse_fraction: float = 1.0,
    sparse_policy: str = "random",
    step: int = 0,
    analyze: bool = False,
    # --- new parameters (forwarded to perturbation_gradients) ---
    layer_fractions=None,
    probe_layer: int | None = None,
    log_coverage: bool = False,
    coverage_threshold: float = 1e-12,
    noise_distribution: str = "gaussian",
    noise_sampling: str = "single",
    # --- normalizer ablation ---
    norm_mode: str = "exact",
    norm_beta: float = 0.99,
    norm_update_every: int = 1,
    norm_eps: float = 1e-8,
    norm_state: dict | None = None,
    # --- linearized ANP ---
    delta_mode: str = "noisy",
    # --- lazy decorrelation (DANP only) ---
    decor_update_every: int = 1,
):
    sparse_info = None
    grad_dist = None
    coverage_stats = None
    compare_diag = None
    decor_update_performed = False

    # delta_mode restrictions
    if delta_mode != "noisy":
        if decorrelated:
            raise NotImplementedError("delta_mode != 'noisy' is not supported for DANP")
        if algorithm not in {"anp", "np"}:
            raise NotImplementedError(
                f"delta_mode != 'noisy' is only supported for algorithm in {{anp, np}}, "
                f"got {algorithm!r}"
            )
        if noise_sampling != "single":
            raise NotImplementedError(
                "delta_mode != 'noisy' is only supported with noise_sampling='single'"
            )

    sal_cost = None
    if algorithm == "bp":
        grads, y_pred, loss_per_sample = bp_gradients(
            model=model,
            x=x,
            y=y,
            loss_fn=loss_fn,
            decorrelated=decorrelated,
        )
    elif algorithm in {"np", "anp", "inp"}:
        grads, y_pred, loss_per_sample, sparse_info, grad_dist, coverage_stats, compare_diag, sal_cost = \
            perturbation_gradients(
                model=model,
                x=x,
                y=y,
                loss_fn=loss_fn,
                decorrelated=decorrelated,
                variant=algorithm,
                noise_std=noise_std,
                num_noise_iters=num_noise_iters,
                sparse=sparse,
                sparse_fraction=sparse_fraction,
                sparse_policy=sparse_policy,
                step=step,
                analyze=analyze,
                layer_fractions=layer_fractions,
                probe_layer=probe_layer,
                log_coverage=log_coverage,
                coverage_threshold=coverage_threshold,
                noise_distribution=noise_distribution,
                noise_sampling=noise_sampling,
                norm_mode=norm_mode,
                norm_beta=norm_beta,
                norm_update_every=norm_update_every,
                norm_eps=norm_eps,
                norm_state=norm_state,
                delta_mode=delta_mode,
            )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Lazy decorrelation: R is APPLIED every step inside model.forward()/forward_noisy()
    # (decorrelate_inputs uses the current R unconditionally, unchanged above). Only the
    # expensive R <- R - decor_lr * C~R UPDATE itself is gated here, by global step count.
    # This does not remove R application from the forward/update path, and it does not
    # change the main ANP/DANP weight update frequency (optimizer.apply_gradients below
    # still runs every step).
    if decorrelated:
        if step % decor_update_every == 0:
            apply_decorrelation_update(model, decor_lr)
            decor_update_performed = True

    optimizer.apply_gradients(zip(grads, model.ordered_trainable_variables()))
    return (y_pred, loss_per_sample, sparse_info, grad_dist, coverage_stats, compare_diag,
            decor_update_performed, sal_cost)