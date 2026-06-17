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
)
from analysis import compute_gradient_dist_stats


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


def _np_like_grads_from_cached_pass(
    model,
    performance_diff: tf.Tensor,
    variant: str,
    layer_idx: int | None = None,
    masks: list[tf.Tensor] | None = None,
) -> list[tf.Tensor]:
    """
    variant in {"np", "anp", "inp"}.
    Returns grads aligned with model.ordered_trainable_variables().

    masks: optional list of per-layer masks with shape [1, units] used by the
    sparse NP/ANP/DANP variants. For "np", layer.noise is already zeroed for
    unselected nodes (see model.reset_all_noise), so error is automatically
    sparse. For "anp", activity_diff is masked here and the normalization in
    _network_activity_stats is restricted to the selected nodes.
    """
    if variant in {"anp", "np"}:
        network_norm_sq, n_total = _network_activity_stats(model, masks=masks)

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
            error = error / network_norm_sq
            error = error * tf.cast(n_total, error.dtype)

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
):
    """
    variant in {"np", "anp", "inp"}.
    Averages gradients across multiple noisy iterations.

    sparse: when True (only supported for variant in {"np", "anp"}), only a
    fraction `sparse_fraction` of nodes per layer are perturbed AND
    contribute to the update (see sparse.py and model.reset_all_noise).
    Returns an extra `sparse_info` dict (or None if sparse is disabled) with
    keys "n_active", "n_total", "fraction" describing the last iteration's
    masks, for logging.
    """
    # Clean pass first
    y_clean = model.forward(x, decorrelate=decorrelated)
    loss_clean = loss_fn(y_clean, y)

    all_iter_grads = []
    sparse_info = None
    grad_dist = None
    use_sparse = sparse and variant in {"np", "anp"}

    for it in range(num_noise_iters):
        masks = None

        if use_sparse:
            if sparse_policy == "activity_diff_topk":
                # Two-pass: provisional full-noise forward → top-k by mean|Δact| → mask noise.
                model.reset_all_noise(noise_std, masks=None)
                _ = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
                masks = compute_activity_diff_topk_masks(model, fraction=sparse_fraction)
                score_stats = compute_activity_diff_score_stats(model, masks)
                for layer, mask in zip(model.layers_list, masks):
                    layer.noise = layer.noise * tf.cast(mask, layer.noise.dtype)
                    layer.noise_mask = mask

            elif sparse_policy == "activity_loss_topk":
                # Two-pass: provisional forward → top-k by mean|Δact × Δloss| → mask noise.
                model.reset_all_noise(noise_std, masks=None)
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
                # (full ANP effective-error magnitude g_i) → mask noise.
                model.reset_all_noise(noise_std, masks=None)
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
                masks = compute_sparse_masks(
                    model,
                    fraction=sparse_fraction,
                    policy=sparse_policy,
                    step=step * num_noise_iters + it,
                )
                score_stats = {
                    "score_selected_mean": float("nan"),
                    "score_unselected_mean": float("nan"),
                }
                model.reset_all_noise(noise_std, masks=masks)

            base_stats = mask_stats(model, masks)
            sparse_info = {**base_stats, "policy": sparse_policy, **score_stats}
        else:
            model.reset_all_noise(noise_std, masks=None)

        if variant in {"np", "anp"}:
            y_noisy = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
            loss_noisy = loss_fn(y_noisy, y)
            performance_diff = tf.reshape(loss_clean - loss_noisy, [-1, 1])

            # Analysis uses raw loss-based performance_diff, before any variant-
            # specific scaling (e.g. NP's 1/σ² factor), so that g_i estimates
            # the true ANP numerator regardless of which variant is training.
            if analyze and it == num_noise_iters - 1:
                grad_dist = compute_gradient_dist_stats(model, performance_diff)

            if variant == "np":
                performance_diff = performance_diff / tf.cast(noise_std**2, performance_diff.dtype)

            grads = _np_like_grads_from_cached_pass(
                model=model,
                performance_diff=performance_diff,
                variant=variant,
                layer_idx=None,
                masks=masks,
            )

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
    return mean_grads, y_clean, loss_clean, sparse_info, grad_dist


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
):
    sparse_info = None
    grad_dist = None

    if algorithm == "bp":
        grads, y_pred, loss_per_sample = bp_gradients(
            model=model,
            x=x,
            y=y,
            loss_fn=loss_fn,
            decorrelated=decorrelated,
        )
    elif algorithm in {"np", "anp", "inp"}:
        grads, y_pred, loss_per_sample, sparse_info, grad_dist = perturbation_gradients(
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
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if decorrelated:
        apply_decorrelation_update(model, decor_lr)

    optimizer.apply_gradients(zip(grads, model.ordered_trainable_variables()))
    return y_pred, loss_per_sample, sparse_info, grad_dist