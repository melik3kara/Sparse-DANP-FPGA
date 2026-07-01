"""
Sparse node-selection policies for Node Perturbation (NP) and Activity-based
Node Perturbation (ANP).

All policies produce a binary mask of shape [1, units] per layer.  The mask is
applied to layer noise before the noisy forward pass, so unselected nodes
receive zero noise AND contribute zero gradient — giving a sparse perturbation
AND a sparse weight update at cost proportional to the active fraction k/H.

Policy overview
---------------
random               Uniformly random k-subset each iteration.  Acts as an
                     unbiased gradient estimator and provides dropout-like
                     regularisation.  Empirically the strongest baseline.

scheduled            Deterministic round-robin: each step shifts the active
                     window by k, guaranteeing full coverage over H/k steps.
                     Useful for FPGA schedules where true randomness is costly.

activation_threshold Top-k by mean |clean activation|.  Selects nodes that are
                     most active in the clean forward pass.

activity_diff_topk   Top-k by mean |Δactivation|, estimated from a provisional
                     full-noise forward pass.  Ranks by response magnitude but
                     ignores whether the response reduced the loss.

activity_loss_topk   Top-k by mean |Δactivation × Δloss|.  Scores the ANP
                     numerator directly, so rank reflects loss-correlated
                     response.

gradient_aligned_topk
                     Top-k by mean |Δactivation × Δloss / ‖Δact‖²|.  Scores
                     the full per-node ANP effective-error magnitude g_i,
                     accounting for the network-level normalisation.  The most
                     theoretically aligned policy with the ANP update.
"""
from __future__ import annotations

from typing import Union
import tensorflow as tf


# ---------------------------------------------------------------------------
# Layer allocation
# ---------------------------------------------------------------------------

def allocate_fractions(
    layer_sizes: list[int],
    total_fraction: float,
    mode: str,
) -> list[float]:
    """Return per-layer active fractions that maintain the total perturbation budget.

    The total active nodes across all layers is approximately
    ``total_fraction * sum(layer_sizes)``, distributed according to ``mode``.

    Modes
    -----
    uniform       Same fraction for all layers — existing behaviour.
    front_loaded  Earlier layers receive proportionally more of the budget.
    back_loaded   Later layers receive proportionally more of the budget.
    middle_loaded Middle layers receive more of the budget; falls back to
                  uniform when the network has fewer than 3 layers.

    Note: fractions are clipped to [1/H_l, 1.0] per layer, so the total
    active-node count may be slightly less than ``total_fraction * Σ H_l``
    when a small layer (e.g. a 10-node output layer) hits its ceiling.
    """
    L = len(layer_sizes)
    if L == 0:
        return []
    if mode == "uniform" or L == 1:
        return [total_fraction] * L

    n_total = sum(layer_sizes)
    k_total = max(L, round(total_fraction * n_total))

    if mode == "front_loaded":
        weights = [float(L - i) for i in range(L)]
    elif mode == "back_loaded":
        weights = [float(i + 1) for i in range(L)]
    elif mode == "middle_loaded":
        if L < 3:
            return [total_fraction] * L
        weights = [float(1 + min(i, L - 1 - i)) for i in range(L)]
    else:
        raise ValueError(
            f"Unknown layer_allocation mode '{mode}'. "
            "Choose from: uniform, front_loaded, back_loaded, middle_loaded."
        )

    w_sum = sum(weights)
    fracs: list[float] = []
    allocated = 0
    for i, (w, h) in enumerate(zip(weights, layer_sizes)):
        if i == L - 1:
            # Last layer absorbs any rounding remainder.
            k_l = max(1, min(h, k_total - allocated))
        else:
            k_l = max(1, min(h, round(k_total * w / w_sum)))
        allocated += k_l
        fracs.append(k_l / h)
    return fracs


def num_active_units(units: int, fraction: float) -> int:
    """Number of active nodes k for a layer with H=units, given fraction k/H.

    Always at least 1 and at most `units`.
    """
    k = int(round(units * fraction))
    return max(1, min(units, k))


def random_mask(units: int, fraction: float, dtype=tf.float32) -> tf.Tensor:
    """Uniformly random subset of `k` active nodes. Shape [1, units]."""
    k = num_active_units(units, fraction)
    perm = tf.random.shuffle(tf.range(units))
    idx = perm[:k]
    mask = tf.scatter_nd(tf.expand_dims(idx, axis=1), tf.ones([k], dtype=dtype), [units])
    return tf.reshape(mask, [1, units])


def scheduled_mask(units: int, fraction: float, step: int, dtype=tf.float32) -> tf.Tensor:
    """Deterministic round-robin subset of `k` active nodes, shifting with `step`.

    Guarantees full coverage of all nodes over time, which is useful for
    FPGA-style round-robin perturbation schedules. Shape [1, units].
    """
    k = num_active_units(units, fraction)
    start = (step * k) % units
    idx = tf.math.floormod(tf.range(start, start + k), units)
    mask = tf.scatter_nd(tf.expand_dims(idx, axis=1), tf.ones([k], dtype=dtype), [units])
    return tf.reshape(mask, [1, units])


def activation_threshold_mask(activations: tf.Tensor, fraction: float) -> tf.Tensor:
    """Top-k nodes by mean |activation| over the batch. Shape [1, units].

    `activations` has shape [batch, units] (e.g. layer.outputs_clean).
    """
    units = activations.shape[-1]
    k = num_active_units(units, fraction)
    score = tf.reduce_mean(tf.abs(activations), axis=0)  # [units]
    _, idx = tf.math.top_k(score, k=k)
    mask = tf.scatter_nd(
        tf.expand_dims(idx, axis=1), tf.ones([k], dtype=activations.dtype), [units]
    )
    return tf.reshape(mask, [1, units])


def compute_sparse_masks(
    model,
    fraction: Union[float, list],
    policy: str,
    step: int = 0,
) -> list[tf.Tensor]:
    """Compute one mask of shape [1, units] per layer in `model.layers_list`.

    Parameters
    ----------
    fraction : float or list[float]
        A single fraction applied uniformly to all layers, **or** a list of
        per-layer fractions (one per entry in ``model.layers_list``).
        Per-layer fractions are produced by :func:`allocate_fractions`.
    policy : str
        One of ``"random"``, ``"scheduled"``, ``"activation_threshold"``.
    step : int
        Global training step, used by the ``"scheduled"`` policy.

    Requires a clean forward pass to have been run already (``layer.outputs_clean``
    must be populated) for the ``"activation_threshold"`` policy.
    """
    masks = []
    for i, layer in enumerate(model.layers_list):
        f = fraction[i] if isinstance(fraction, list) else fraction
        units = layer.units

        if policy == "random":
            mask = random_mask(units, f)
        elif policy == "scheduled":
            mask = scheduled_mask(units, f, step=step)
        elif policy == "activation_threshold":
            if layer.outputs_clean is None:
                raise RuntimeError(
                    "Need a clean forward pass before computing activation_threshold masks."
                )
            mask = activation_threshold_mask(layer.outputs_clean, f)
        else:
            raise ValueError(f"Unknown sparse policy: {policy}")

        masks.append(mask)
    return masks


def build_probe_masks(
    model,
    probe_layer: int,
    probe_fraction: float,
    policy: str,
    step: int = 0,
) -> list[tf.Tensor]:
    """Build masks for single-layer probe mode.

    The probe layer receives a policy-based mask with ``probe_fraction`` active
    nodes.  Every other layer receives an all-zeros mask (zero noise, zero
    gradient).

    ``policy`` must be one of the one-pass policies: ``"random"``,
    ``"scheduled"``, or ``"activation_threshold"``.  Two-pass adaptive policies
    (``activity_diff_topk`` etc.) are not supported in probe mode because they
    require a full-network provisional forward pass.
    """
    masks = []
    for i, layer in enumerate(model.layers_list):
        if i != probe_layer:
            masks.append(tf.zeros([1, layer.units], dtype=tf.float32))
        elif policy == "random":
            masks.append(random_mask(layer.units, probe_fraction))
        elif policy == "scheduled":
            masks.append(scheduled_mask(layer.units, probe_fraction, step=step))
        elif policy == "activation_threshold":
            if layer.outputs_clean is None:
                raise RuntimeError(
                    "Need a clean forward pass before computing activation_threshold mask."
                )
            masks.append(activation_threshold_mask(layer.outputs_clean, probe_fraction))
        else:
            raise ValueError(
                f"--probe_layer does not support two-pass policy '{policy}'. "
                "Use random, scheduled, or activation_threshold."
            )
    return masks


def mask_stats(model, masks: list[tf.Tensor]) -> dict:
    """Summarize active-node counts across the whole network.

    Returns a dict with:
      - n_active: total number of perturbed/updated nodes across all layers
      - n_total: total number of nodes across all layers
      - fraction: n_active / n_total (relative perturbation/update cost)
    """
    n_active = 0
    n_total = 0
    for layer, mask in zip(model.layers_list, masks):
        units = layer.units
        active = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())
        n_active += active
        n_total += units

    fraction = (n_active / n_total) if n_total > 0 else 0.0
    return {"n_active": n_active, "n_total": n_total, "fraction": fraction}


# ---------------------------------------------------------------------------
# activity_diff_topk policy
# ---------------------------------------------------------------------------

def activity_diff_topk_mask(activity_diff: tf.Tensor, fraction: float) -> tf.Tensor:
    """Select the top-k nodes per layer by mean |activity_diff| over the batch.

    activity_diff: [batch, units] = layer.outputs_noisy - layer.outputs_clean
    computed from a **provisional full-noise forward pass** (before masking).

    Returns [1, units] float mask.
    Falls back to a uniformly random mask when all scores are zero (e.g. dead
    neurons), so the layer always participates with at least one active node.
    """
    units = activity_diff.shape[-1]
    k = num_active_units(units, fraction)
    scores = tf.reduce_mean(tf.abs(activity_diff), axis=0)  # [units]

    if float(tf.reduce_max(scores).numpy()) == 0.0:
        return random_mask(units, fraction, dtype=activity_diff.dtype)

    _, idx = tf.math.top_k(scores, k=k)
    mask = tf.scatter_nd(
        tf.expand_dims(idx, axis=1), tf.ones([k], dtype=activity_diff.dtype), [units]
    )
    return tf.reshape(mask, [1, units])


def compute_activity_diff_topk_masks(model, fraction: float) -> list[tf.Tensor]:
    """Build one activity-diff top-k mask per layer.

    Requires BOTH layer.outputs_clean and layer.outputs_noisy to be populated,
    i.e. a clean forward pass and a provisional full-noise forward pass must
    have been run first (before applying any mask to layer.noise).
    """
    masks = []
    for layer in model.layers_list:
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            raise RuntimeError(
                "Need a clean AND a provisional noisy forward pass before "
                "computing activity_diff_topk masks."
            )
        activity_diff = layer.outputs_noisy - layer.outputs_clean
        masks.append(activity_diff_topk_mask(activity_diff, fraction))
    return masks


def compute_activity_diff_score_stats(model, masks: list[tf.Tensor]) -> dict:
    """Mean activity-diff score of selected vs unselected nodes across all layers.

    Must be called right after compute_activity_diff_topk_masks and BEFORE the
    provisional noise in layer.noise is masked out, because this function reads
    layer.outputs_noisy which still holds full-noise activations at that point.

    Returns:
      - score_selected_mean: mean score (mean|activity_diff|) over active nodes
      - score_unselected_mean: mean score over inactive nodes
    """
    selected_scores = []
    unselected_scores = []

    for layer, mask in zip(model.layers_list, masks):
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            continue
        activity_diff = layer.outputs_noisy - layer.outputs_clean
        scores = tf.reduce_mean(tf.abs(activity_diff), axis=0)  # [units]
        mask_1d = tf.cast(tf.reshape(mask, [-1]), tf.bool)      # [units]

        sel = tf.boolean_mask(scores, mask_1d)
        unsel = tf.boolean_mask(scores, ~mask_1d)

        if int(tf.size(sel).numpy()) > 0:
            selected_scores.append(float(tf.reduce_mean(sel).numpy()))
        if int(tf.size(unsel).numpy()) > 0:
            unselected_scores.append(float(tf.reduce_mean(unsel).numpy()))

    sel_mean = sum(selected_scores) / len(selected_scores) if selected_scores else float("nan")
    unsel_mean = sum(unselected_scores) / len(unselected_scores) if unselected_scores else float("nan")
    return {"score_selected_mean": sel_mean, "score_unselected_mean": unsel_mean}


# ---------------------------------------------------------------------------
# activity_loss_topk policy
# ---------------------------------------------------------------------------

def activity_loss_topk_mask(
    activity_diff: tf.Tensor,
    performance_diff: tf.Tensor,
    fraction: float,
) -> tf.Tensor:
    """Select top-k nodes by mean |activity_diff * performance_diff| over the batch.

    activity_diff:    [batch, units]  = layer.outputs_noisy - layer.outputs_clean
    performance_diff: [batch, 1]      = loss_clean - loss_noisy_provisional

    The product activity_diff * performance_diff is the per-node numerator of the
    ANP gradient estimator.  Ranking by its magnitude selects nodes whose
    perturbation response is correlated with actual loss improvement, which is
    more directly aligned with the ANP update than ranking by |activity_diff|
    alone (which ignores whether the response reduced the loss).

    Falls back to random_mask when all scores are zero.
    Returns [1, units] float mask.
    """
    units = activity_diff.shape[-1]
    k = num_active_units(units, fraction)
    scores = tf.reduce_mean(tf.abs(activity_diff * performance_diff), axis=0)  # [units]

    if float(tf.reduce_max(scores).numpy()) == 0.0:
        return random_mask(units, fraction, dtype=activity_diff.dtype)

    _, idx = tf.math.top_k(scores, k=k)
    mask = tf.scatter_nd(
        tf.expand_dims(idx, axis=1), tf.ones([k], dtype=activity_diff.dtype), [units]
    )
    return tf.reshape(mask, [1, units])


def compute_activity_loss_topk_masks(
    model, performance_diff: tf.Tensor, fraction: float
) -> list[tf.Tensor]:
    """Build one activity-loss top-k mask per layer.

    Requires a clean forward pass (outputs_clean) and a provisional full-noise
    forward pass (outputs_noisy) to have been completed first.

    performance_diff: [batch, 1] = loss_clean - loss_noisy_provisional,
    computed from the same provisional forward pass.
    """
    masks = []
    for layer in model.layers_list:
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            raise RuntimeError(
                "Need a clean AND a provisional noisy forward pass before "
                "computing activity_loss_topk masks."
            )
        activity_diff = layer.outputs_noisy - layer.outputs_clean
        masks.append(activity_loss_topk_mask(activity_diff, performance_diff, fraction))
    return masks


def compute_activity_loss_score_stats(
    model, masks: list[tf.Tensor], performance_diff: tf.Tensor
) -> dict:
    """Mean |activity_diff * performance_diff| score of selected vs unselected nodes.

    Call after compute_activity_loss_topk_masks and BEFORE the provisional noise
    is zeroed (layer.outputs_noisy still reflects the full-noise forward pass).

    Returns:
      - score_selected_mean:   mean ANP-numerator score over active nodes
      - score_unselected_mean: mean ANP-numerator score over inactive nodes
    """
    selected_scores = []
    unselected_scores = []

    for layer, mask in zip(model.layers_list, masks):
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            continue
        activity_diff = layer.outputs_noisy - layer.outputs_clean
        scores = tf.reduce_mean(tf.abs(activity_diff * performance_diff), axis=0)  # [units]
        mask_1d = tf.cast(tf.reshape(mask, [-1]), tf.bool)  # [units]

        sel = tf.boolean_mask(scores, mask_1d)
        unsel = tf.boolean_mask(scores, ~mask_1d)

        if int(tf.size(sel).numpy()) > 0:
            selected_scores.append(float(tf.reduce_mean(sel).numpy()))
        if int(tf.size(unsel).numpy()) > 0:
            unselected_scores.append(float(tf.reduce_mean(unsel).numpy()))

    sel_mean = sum(selected_scores) / len(selected_scores) if selected_scores else float("nan")
    unsel_mean = sum(unselected_scores) / len(unselected_scores) if unselected_scores else float("nan")
    return {"score_selected_mean": sel_mean, "score_unselected_mean": unsel_mean}


# ---------------------------------------------------------------------------
# gradient_aligned_topk policy
# ---------------------------------------------------------------------------

def gradient_aligned_topk_mask(
    activity_diff: tf.Tensor,     # [batch, units]
    performance_diff: tf.Tensor,  # [batch, 1]
    network_norm_sq: tf.Tensor,   # [batch, 1]
    fraction: float,
) -> tf.Tensor:
    """Select top-k nodes by mean |activity_diff * performance_diff / network_norm_sq|.

    This score directly estimates the per-node ANP effective-error magnitude:

        g_i ∝ activity_diff_i * performance_diff / ||activity_diff||²

    Ranking by |g_i| selects nodes that contribute most to the actual ANP
    weight update, accounting for both loss-correlation (performance_diff) and
    the full-network normalisation (network_norm_sq).

    All three tensors come from the same provisional full-noise forward pass.
    Falls back to random_mask when all scores are zero or NaN (e.g. degenerate
    norm or dead layer).  Returns [1, units] float mask.
    """
    units = activity_diff.shape[-1]
    k = num_active_units(units, fraction)

    # [batch, units] * [batch, 1] / [batch, 1] → [batch, units] → [units]
    scores = tf.reduce_mean(
        tf.abs(activity_diff * performance_diff / network_norm_sq), axis=0
    )

    max_score = float(tf.reduce_max(scores).numpy())
    if max_score == 0.0 or max_score != max_score:  # zero or NaN guard
        return random_mask(units, fraction, dtype=activity_diff.dtype)

    _, idx = tf.math.top_k(scores, k=k)
    mask = tf.scatter_nd(
        tf.expand_dims(idx, axis=1), tf.ones([k], dtype=activity_diff.dtype), [units]
    )
    return tf.reshape(mask, [1, units])


def compute_gradient_aligned_topk_masks(
    model,
    performance_diff: tf.Tensor,  # [batch, 1]
    network_norm_sq: tf.Tensor,   # [batch, 1]
    fraction: float,
) -> list[tf.Tensor]:
    """Build one gradient-aligned top-k mask per layer.

    Requires a clean forward pass (outputs_clean) and a provisional full-noise
    forward pass (outputs_noisy) to have been completed first.

    performance_diff: [batch, 1] = loss_clean - loss_noisy_provisional
    network_norm_sq:  [batch, 1] = per-sample ||concat(all layer activity_diffs)||²
                      from _network_activity_stats(model, masks=None).
    """
    masks = []
    for layer in model.layers_list:
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            raise RuntimeError(
                "Need a clean AND a provisional noisy forward pass before "
                "computing gradient_aligned_topk masks."
            )
        activity_diff = layer.outputs_noisy - layer.outputs_clean
        masks.append(
            gradient_aligned_topk_mask(
                activity_diff, performance_diff, network_norm_sq, fraction
            )
        )
    return masks


def compute_gradient_aligned_score_stats(
    model,
    masks: list[tf.Tensor],
    performance_diff: tf.Tensor,  # [batch, 1]
    network_norm_sq: tf.Tensor,   # [batch, 1]
) -> dict:
    """Mean |activity_diff * performance_diff / network_norm_sq| for selected vs unselected nodes.

    Call after compute_gradient_aligned_topk_masks and BEFORE the provisional
    noise is zeroed (layer.outputs_noisy still holds the full-noise activations).
    """
    selected_scores = []
    unselected_scores = []

    for layer, mask in zip(model.layers_list, masks):
        if layer.outputs_clean is None or layer.outputs_noisy is None:
            continue
        activity_diff = layer.outputs_noisy - layer.outputs_clean
        scores = tf.reduce_mean(
            tf.abs(activity_diff * performance_diff / network_norm_sq), axis=0
        )  # [units]
        mask_1d = tf.cast(tf.reshape(mask, [-1]), tf.bool)  # [units]

        sel = tf.boolean_mask(scores, mask_1d)
        unsel = tf.boolean_mask(scores, ~mask_1d)

        if int(tf.size(sel).numpy()) > 0:
            selected_scores.append(float(tf.reduce_mean(sel).numpy()))
        if int(tf.size(unsel).numpy()) > 0:
            unselected_scores.append(float(tf.reduce_mean(unsel).numpy()))

    sel_mean = sum(selected_scores) / len(selected_scores) if selected_scores else float("nan")
    unsel_mean = sum(unselected_scores) / len(unselected_scores) if unselected_scores else float("nan")
    return {"score_selected_mean": sel_mean, "score_unselected_mean": unsel_mean}
