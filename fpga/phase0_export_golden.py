#!/usr/bin/env python3
"""
Phase-0 golden-vector exporter for the FPGA A-vs-B milestone.

Debug network: 16 -> 8 -> 4 -> 2  (hidden = leaky_relu alpha=0.2, output = softmax)
Batch size 1, plain ANP (decorrelation disabled).

This script DOES NOT modify any existing repository code. It imports the
repository model (models.MLP / DecorrelatedDense) and the repository linearized
ANP implementation (algorithms.linearized_activity_diffs) and treats them as the
source of truth. The only "adapter" is assigning deterministic weights/biases and
deterministic per-layer epsilon tensors onto the existing model objects, and
reading the cached pre-/post-activations the repository already stores.

Terminology (kept precise):
  - logits            = pre-softmax output-layer values (== outputs_clean/noisy of last layer)
  - softmax outputs   = probabilities (NOT logits)
  - activity_diff     = pre-activation difference delta_z = outputs_noisy - outputs_clean
                        (outputs_* are PRE-activations because DecorrelatedDense has activation=None)

Two exported paths:
  A. Original noisy-forward ANP : full nonlinear noisy forward (model.forward_noisy)
  B. Dense linearized ANP       : first-order recurrence (algorithms.linearized_activity_diffs)

Outputs: manifest.json, golden.npz, and raw little-endian float32 (<f4) binaries.
Row-major (C order) flattening; weight layout is [in, units] (z = x @ W + b).
"""
from __future__ import annotations

import os
import sys
import json
import random
import hashlib
import argparse

# Deterministic / CPU-only before importing TF.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

# Make the repository root importable without changing cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tensorflow as tf

# Repository source of truth (NOT reimplemented here).
from models import MLP
from algorithms import linearized_activity_diffs, _activation_derivative_elementwise

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIM = 16
HIDDEN_SIZES = [8, 4]
OUTPUT_DIM = 2
LAYER_UNITS = HIDDEN_SIZES + [OUTPUT_DIM]     # [8, 4, 2]
DIMS = [INPUT_DIM] + LAYER_UNITS              # [16, 8, 4, 2]
N_LAYERS = len(LAYER_UNITS)                   # 3
ALPHA = 0.2                                   # leaky_relu negative slope
BATCH = 1
DTYPE_STR = "<f4"                             # little-endian float32

# Acceptance constraints so both leaky-relu branches are exercised and the
# activation derivative is unambiguous (no hidden pre-activation near zero).
MIN_ABS_HIDDEN_Z = 1e-3
SEED_SEARCH = range(0, 1000)                  # first passing seed is chosen (deterministic)

# Fixture registry: name -> output directory (all under fpga/phase0/).
FIXTURES = {
    "nominal":         "golden_debug_16_8_4_2",
    "branch_crossing": "golden_stress_branch_crossing_16_8_4_2",
}

# Branch-crossing fixture construction constants (deterministic).
BRANCH_SEED         = 1       # base RNG seed for the stress fixture
CROSS_LAYER         = 0       # first hidden layer (fed by the unperturbed input)
CROSS_NODE          = 0       # selected hidden node
CROSS_TARGET_ZCLEAN = 1e-3    # forced clean pre-activation (small, positive branch)
CROSS_EPSILON       = -1e-2   # opposite-sign noise (10x) -> noisy crosses to negative
BRANCH_MIN_AVSB     = 1e-4    # A-vs-B delta_z diff must exceed this for the stress fixture

# Output directories are resolved per-fixture in main().
OUT_DIR = None
BIN_DIR = None

FAIL_TOL = 1e-5   # max-abs tolerance for consistency self-checks


def leaky(z):
    """Hidden activation, alpha explicitly 0.2 (matches tf.nn.leaky_relu default)."""
    return tf.nn.leaky_relu(z, alpha=ALPHA)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_deterministic_arrays(rng: np.random.Generator):
    """Explicit float32 weights/biases/input/epsilon (not framework initializers)."""
    weights, biases, epsilons = [], [], []
    x = rng.uniform(-1.0, 1.0, size=(BATCH, INPUT_DIM)).astype(np.float32)
    for l in range(N_LAYERS):
        in_dim = DIMS[l]
        units = DIMS[l + 1]
        W = rng.uniform(-0.6, 0.6, size=(in_dim, units)).astype(np.float32)   # layout [in, units]
        b = rng.uniform(-0.3, 0.3, size=(units,)).astype(np.float32)
        eps = rng.uniform(-0.1, 0.1, size=(BATCH, units)).astype(np.float32)  # pre-activation noise
        weights.append(W)
        biases.append(b)
        epsilons.append(eps)
    return x, weights, biases, epsilons


def assign_into_model(model: MLP, weights, biases, epsilons):
    for l, layer in enumerate(model.layers_list):
        layer.kernel.assign(tf.constant(weights[l], dtype=tf.float32))
        layer.bias.assign(tf.constant(biases[l], dtype=tf.float32))
        # Deterministic epsilon and an all-ones mask (dense baseline).
        layer.noise = tf.constant(epsilons[l], dtype=tf.float32)
        layer.noise_mask = tf.ones([1, layer.units], dtype=tf.float32)


def hidden_z_ok(z_clean_list) -> bool:
    """Both signs present and no near-zero entry in every hidden layer."""
    for l in range(N_LAYERS - 1):  # hidden layers only
        z = z_clean_list[l].reshape(-1)
        if z.min() >= 0.0 or z.max() <= 0.0:
            return False
        if np.min(np.abs(z)) <= MIN_ABS_HIDDEN_Z:
            return False
    return True


def run_repository_paths(model: MLP, x_np: np.ndarray):
    """Drive the repository implementations and read back cached tensors."""
    x = tf.constant(x_np, dtype=tf.float32)

    # --- Clean forward (repository) : sets layer.outputs_clean (pre-activations) ---
    clean_probs = model.forward(x, decorrelate=False)            # softmax probabilities
    z_clean = [layer.outputs_clean.numpy() for layer in model.layers_list]
    a_clean = [leaky(model.layers_list[l].outputs_clean).numpy()
               if l < N_LAYERS - 1
               else tf.nn.softmax(model.layers_list[l].outputs_clean).numpy()
               for l in range(N_LAYERS)]
    clean_logits = z_clean[-1]                                    # pre-softmax output values

    # --- A. Original noisy-forward ANP (repository) ---
    # forward_noisy adds layer.noise at every layer (noise_layer_idx=None) and
    # propagates the perturbation nonlinearly. It does NOT touch outputs_clean.
    noisy_probs = model.forward_noisy(x, decorrelate=False, noise_layer_idx=None)
    z_noisy = [layer.outputs_noisy.numpy() for layer in model.layers_list]
    a_noisy = [leaky(model.layers_list[l].outputs_noisy).numpy()
               if l < N_LAYERS - 1
               else tf.nn.softmax(model.layers_list[l].outputs_noisy).numpy()
               for l in range(N_LAYERS)]
    noisy_logits = z_noisy[-1]
    dz_original = [z_noisy[l] - z_clean[l] for l in range(N_LAYERS)]

    # --- B. Dense linearized ANP (repository) ---
    # Reads outputs_clean + layer.noise (same epsilon). Independent of the noisy pass above.
    delta_z_list, y_noisy_approx = linearized_activity_diffs(model)
    dz_linearized = [dz.numpy() for dz in delta_z_list]
    # Hidden-layer delta_a = f'(z_clean) * delta_z, using the repository derivative helper.
    da_linearized = []
    for l in range(N_LAYERS - 1):
        fprime = _activation_derivative_elementwise(
            model.layers_list[l].activation_fn, model.layers_list[l].outputs_clean
        )
        da_linearized.append((fprime * delta_z_list[l]).numpy())
    approx_logits = (model.layers_list[-1].outputs_clean + delta_z_list[-1]).numpy()
    approx_probs = y_noisy_approx.numpy()

    return dict(
        clean_probs=clean_probs.numpy(),
        noisy_probs=noisy_probs.numpy(),
        z_clean=z_clean, a_clean=a_clean,
        z_noisy=z_noisy, a_noisy=a_noisy,
        clean_logits=clean_logits, noisy_logits=noisy_logits,
        dz_original=dz_original,
        dz_linearized=dz_linearized, da_linearized=da_linearized,
        approx_logits=approx_logits, approx_probs=approx_probs,
    )


def independent_linearized_recompute(weights, biases, epsilons, z_clean):
    """
    Verification-only numpy recomputation of the linearized recurrence from the
    EXPORTED kernel inputs (weights, epsilon) and clean pre-activations. Used
    solely to confirm the exported binaries reconstruct the repository result;
    it is NOT the source of the golden vectors.

        delta_a_0 = 0
        delta_z_l = delta_a_prev @ W_l + eps_l
        delta_a_l = f'(z_clean_l) * delta_z_l   (hidden only), f' = 1 if z>0 else alpha
    """
    dz = []
    da_prev = None
    for l in range(N_LAYERS):
        z = epsilons[l].copy() if da_prev is None else (da_prev @ weights[l] + epsilons[l])
        dz.append(z)
        if l < N_LAYERS - 1:
            fprime = np.where(z_clean[l] > 0.0, 1.0, ALPHA).astype(np.float32)
            da_prev = (fprime * z).astype(np.float32)
    return dz


def independent_original_recompute(x_np, weights, biases, epsilons):
    """
    Verification-only numpy recomputation of the ORIGINAL clean and noisy
    forward passes from the exported inputs. Used to confirm the exported
    binaries reconstruct the repository's original path (not their source).
    """
    a_clean = x_np.astype(np.float32)
    a_noisy = x_np.astype(np.float32)
    z_clean_list, z_noisy_list = [], []
    for l in range(N_LAYERS):
        zc = (a_clean @ weights[l] + biases[l]).astype(np.float32)
        zn = (a_noisy @ weights[l] + biases[l] + epsilons[l]).astype(np.float32)
        z_clean_list.append(zc)
        z_noisy_list.append(zn)
        if l < N_LAYERS - 1:  # leaky_relu on hidden layers only
            a_clean = np.where(zc > 0.0, zc, ALPHA * zc).astype(np.float32)
            a_noisy = np.where(zn > 0.0, zn, ALPHA * zn).astype(np.float32)
    return z_clean_list, z_noisy_list


def _new_model() -> MLP:
    return MLP(
        input_dim=INPUT_DIM,
        hidden_sizes=HIDDEN_SIZES,
        output_dim=OUTPUT_DIM,
        hidden_activation=leaky,
        output_activation=tf.nn.softmax,
    )


def build_nominal_fixture():
    """
    Nominal fixture: first deterministic seed whose hidden pre-activations
    exercise both Leaky-ReLU branches and stay away from zero (unambiguous
    derivative). This preserves the original no-argument behaviour exactly.
    """
    for seed in SEED_SEARCH:
        set_all_seeds(seed)
        rng = np.random.default_rng(seed)
        x_np, weights, biases, epsilons = build_deterministic_arrays(rng)

        model = _new_model()
        assign_into_model(model, weights, biases, epsilons)

        _ = model.forward(tf.constant(x_np, dtype=tf.float32), decorrelate=False)
        z_probe = [layer.outputs_clean.numpy() for layer in model.layers_list]
        if hidden_z_ok(z_probe):
            meta = {
                "chosen_seed": seed,
                "seed_search_range": [SEED_SEARCH.start, SEED_SEARCH.stop],
                "acceptance": {
                    "both_leaky_branches_per_hidden_layer": True,
                    "min_abs_hidden_z": MIN_ABS_HIDDEN_Z,
                },
            }
            return model, x_np, weights, biases, epsilons, meta

    fail("No seed in SEED_SEARCH satisfied the hidden-z acceptance constraints.")


def build_branch_crossing_fixture():
    """
    Stress fixture: deterministically force one hidden Leaky-ReLU node across the
    zero boundary between the clean and noisy paths (z_clean * z_noisy < 0).

    Construction (only weights/bias/epsilon are assigned; no repo code modified):
      1. deterministic random weights/bias/eps from BRANCH_SEED
      2. clean forward to read the selected node's current clean pre-activation
      3. shift that node's bias so its clean pre-activation == CROSS_TARGET_ZCLEAN (+1e-3)
      4. set that node's epsilon == CROSS_EPSILON (-1e-2)

    Because CROSS_LAYER is a hidden layer fed by the unperturbed input, its noisy
    pre-activation is exactly z_clean + epsilon = +1e-3 - 1e-2 = -9e-3, so the
    node lands on the negative branch while its clean value is on the positive
    branch — a genuine branch crossing.
    """
    set_all_seeds(BRANCH_SEED)
    rng = np.random.default_rng(BRANCH_SEED)
    x_np, weights, biases, epsilons = build_deterministic_arrays(rng)

    model = _new_model()
    assign_into_model(model, weights, biases, epsilons)

    # Read the selected node's current clean pre-activation.
    _ = model.forward(tf.constant(x_np, dtype=tf.float32), decorrelate=False)
    z_before = float(model.layers_list[CROSS_LAYER].outputs_clean.numpy()[0, CROSS_NODE])

    # Copy-on-write so we do not mutate the RNG-produced arrays in place.
    biases = [b.copy() for b in biases]
    epsilons = [e.copy() for e in epsilons]
    bias_shift = np.float32(CROSS_TARGET_ZCLEAN) - np.float32(z_before)
    biases[CROSS_LAYER][CROSS_NODE] = (biases[CROSS_LAYER][CROSS_NODE] + bias_shift).astype(np.float32)
    epsilons[CROSS_LAYER][0, CROSS_NODE] = np.float32(CROSS_EPSILON)

    # Re-assign the modified bias and epsilon into the model.
    assign_into_model(model, weights, biases, epsilons)

    meta = {
        "seed": BRANCH_SEED,
        "crossing_construction": {
            "layer_index": CROSS_LAYER,
            "node_index": CROSS_NODE,
            "target_z_clean": float(CROSS_TARGET_ZCLEAN),
            "epsilon": float(CROSS_EPSILON),
            "bias_shift": float(bias_shift),
            "z_clean_before_shift": z_before,
        },
    }
    return model, x_np, weights, biases, epsilons, meta


def branch_label(z: float) -> str:
    return "positive" if z > 0.0 else ("negative" if z < 0.0 else "zero")


def find_branch_crossings(res) -> list:
    """Hidden nodes where z_clean * z_noisy < 0 (Leaky-ReLU branch crossings)."""
    crossings = []
    for l in range(N_LAYERS - 1):  # hidden layers only
        zc = res["z_clean"][l].reshape(-1)
        zn = res["z_noisy"][l].reshape(-1)
        for j in range(zc.shape[0]):
            if float(zc[j]) * float(zn[j]) < 0.0:
                crossings.append((l, j, float(zc[j]), float(zn[j])))
    return crossings


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_bin(name: str, arr: np.ndarray) -> str:
    """Write row-major (C order) little-endian float32; return bin filename."""
    path = os.path.join(BIN_DIR, name)
    a = np.ascontiguousarray(arr, dtype="<f4")
    a.tofile(path)
    return os.path.relpath(path, OUT_DIR)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    global OUT_DIR, BIN_DIR

    parser = argparse.ArgumentParser(
        description="Phase-0 golden-vector exporter (16->8->4->2 debug network)."
    )
    parser.add_argument(
        "--fixture", choices=sorted(FIXTURES.keys()), default="nominal",
        help="Which fixture to generate. 'nominal' (default) preserves the "
             "original no-argument behaviour; 'branch_crossing' forces a hidden "
             "Leaky-ReLU node across zero between the clean and noisy paths.",
    )
    args = parser.parse_args()
    fixture_name = args.fixture

    OUT_DIR = os.path.join(_THIS_DIR, "phase0", FIXTURES[fixture_name])
    BIN_DIR = os.path.join(OUT_DIR, "bin")
    os.makedirs(BIN_DIR, exist_ok=True)

    print(f"[info] fixture = {fixture_name}")
    print(f"[info] output  = {os.path.relpath(OUT_DIR, _REPO_ROOT)}")

    # --- Build the requested deterministic fixture ---
    if fixture_name == "nominal":
        model, x_np, weights, biases, epsilons, build_meta = build_nominal_fixture()
        print(f"[info] chosen deterministic seed = {build_meta['chosen_seed']}")
    else:
        model, x_np, weights, biases, epsilons, build_meta = build_branch_crossing_fixture()
        cc = build_meta["crossing_construction"]
        print(f"[info] branch-crossing seed = {build_meta['seed']}; forcing layer "
              f"{cc['layer_index']} node {cc['node_index']} "
              f"(target z_clean={cc['target_z_clean']:+.3e}, epsilon={cc['epsilon']:+.3e})")

    # --- Drive repository implementations (re-run cleanly on the chosen model) ---
    res = run_repository_paths(model, x_np)

    # --- Detect hidden Leaky-ReLU branch crossings (z_clean * z_noisy < 0) ---
    crossings = find_branch_crossings(res)
    if crossings:
        print(f"[info] hidden branch crossings found: {len(crossings)}")
        for (cl, cj, zc, zn) in crossings:
            print(f"        layer {cl} node {cj}: z_clean={zc:+.6e} ({branch_label(zc)}) "
                  f"-> z_noisy={zn:+.6e} ({branch_label(zn)})  product={zc*zn:+.3e}")

    # ------------------------------------------------------------------ #
    # Self-checks
    # ------------------------------------------------------------------ #
    print("\n=== Self-checks ===")

    # (1) Finiteness of every exported tensor.
    to_check = {
        "input": x_np, "clean_logits": res["clean_logits"],
        "noisy_logits": res["noisy_logits"], "clean_probs": res["clean_probs"],
        "noisy_probs": res["noisy_probs"], "approx_logits": res["approx_logits"],
        "approx_probs": res["approx_probs"],
    }
    for l in range(N_LAYERS):
        to_check[f"W{l}"] = weights[l]; to_check[f"b{l}"] = biases[l]
        to_check[f"eps{l}"] = epsilons[l]
        to_check[f"z_clean_{l}"] = res["z_clean"][l]; to_check[f"z_noisy_{l}"] = res["z_noisy"][l]
        to_check[f"a_clean_{l}"] = res["a_clean"][l]; to_check[f"a_noisy_{l}"] = res["a_noisy"][l]
        to_check[f"dz_original_{l}"] = res["dz_original"][l]
        to_check[f"dz_linearized_{l}"] = res["dz_linearized"][l]
    for l in range(N_LAYERS - 1):
        to_check[f"da_linearized_{l}"] = res["da_linearized"][l]
    for name, arr in to_check.items():
        if not np.all(np.isfinite(arr)):
            fail(f"Non-finite value in tensor '{name}'.")
    print(f"[ok] finiteness: all {len(to_check)} tensors finite")

    # (2) Shape checks.
    expected_shapes = {"input": (BATCH, INPUT_DIM)}
    for l in range(N_LAYERS):
        expected_shapes[f"W{l}"] = (DIMS[l], DIMS[l + 1])
        expected_shapes[f"b{l}"] = (DIMS[l + 1],)
        expected_shapes[f"eps{l}"] = (BATCH, DIMS[l + 1])
        for key in (f"z_clean_{l}", f"z_noisy_{l}", f"a_clean_{l}", f"a_noisy_{l}",
                    f"dz_original_{l}", f"dz_linearized_{l}"):
            expected_shapes[key] = (BATCH, DIMS[l + 1])
    for l in range(N_LAYERS - 1):
        expected_shapes[f"da_linearized_{l}"] = (BATCH, DIMS[l + 1])
    for key in ("clean_logits", "noisy_logits", "clean_probs", "noisy_probs",
                "approx_logits", "approx_probs"):
        expected_shapes[key] = (BATCH, OUTPUT_DIM)
    for name, shp in expected_shapes.items():
        got = tuple(np.asarray(to_check[name]).shape)
        if got != shp:
            fail(f"Shape mismatch for '{name}': got {got}, expected {shp}")
    print(f"[ok] shapes: all {len(expected_shapes)} tensors match expected shapes")

    # (3) Recompute delta_z_original from exported noisy & clean pre-activations.
    err_dz_orig = max(
        float(np.max(np.abs(res["dz_original"][l] - (res["z_noisy"][l] - res["z_clean"][l]))))
        for l in range(N_LAYERS)
    )
    print(f"[ok] delta_z_original == z_noisy - z_clean : max_abs_err = {err_dz_orig:.3e}")
    if err_dz_orig > FAIL_TOL:
        fail(f"delta_z_original inconsistency {err_dz_orig:.3e} > {FAIL_TOL:.1e}")

    # (4) Verify exported inputs reconstruct the repository linearized result.
    dz_lin_np = independent_linearized_recompute(weights, biases, epsilons, res["z_clean"])
    err_dz_lin = max(
        float(np.max(np.abs(dz_lin_np[l] - res["dz_linearized"][l]))) for l in range(N_LAYERS)
    )
    print(f"[ok] linearized recurrence (numpy from exported inputs) vs repository : "
          f"max_abs_err = {err_dz_lin:.3e}")
    if err_dz_lin > FAIL_TOL:
        fail(f"Linearized reconstruction mismatch {err_dz_lin:.3e} > {FAIL_TOL:.1e}")

    # (5) Approx logits / softmax consistency.
    err_approx_logits = float(np.max(np.abs(
        res["approx_logits"] - (res["z_clean"][-1] + res["dz_linearized"][-1]))))
    err_approx_probs = float(np.max(np.abs(
        res["approx_probs"] - tf.nn.softmax(res["approx_logits"]).numpy())))
    print(f"[ok] approx_logits == z_clean_out + dz_lin_out : max_abs_err = {err_approx_logits:.3e}")
    print(f"[ok] approx_probs == softmax(approx_logits)    : max_abs_err = {err_approx_probs:.3e}")
    if max(err_approx_logits, err_approx_probs) > FAIL_TOL:
        fail("Approx logits/probs consistency check failed.")

    # Report the A-vs-B approximation error at the milestone boundary (delta_z + logits).
    err_dz_AvsB = max(
        float(np.max(np.abs(res["dz_original"][l] - res["dz_linearized"][l])))
        for l in range(N_LAYERS)
    )
    err_logits_AvsB = float(np.max(np.abs(res["noisy_logits"] - res["approx_logits"])))
    print(f"\n[info] A-vs-B delta_z  max_abs diff (original vs linearized) = {err_dz_AvsB:.3e}")
    print(f"[info] A-vs-B logits   max_abs diff (noisy vs approx)        = {err_logits_AvsB:.3e}")

    # (6) Independent numpy recomputation of the ORIGINAL forward vs repository.
    zc_np, zn_np = independent_original_recompute(x_np, weights, biases, epsilons)
    err_orig_fwd = max(
        max(float(np.max(np.abs(zc_np[l] - res["z_clean"][l]))),
            float(np.max(np.abs(zn_np[l] - res["z_noisy"][l]))))
        for l in range(N_LAYERS)
    )
    print(f"[ok] original forward (numpy from exported inputs) vs repository : "
          f"max_abs_err = {err_orig_fwd:.3e}")
    if err_orig_fwd > FAIL_TOL:
        fail(f"Original-forward reconstruction mismatch {err_orig_fwd:.3e} > {FAIL_TOL:.1e}")

    # --- Branch-crossing specific self-checks ---
    if fixture_name == "branch_crossing":
        cc = build_meta["crossing_construction"]
        sel_zc = float(res["z_clean"][cc["layer_index"]][0, cc["node_index"]])
        sel_zn = float(res["z_noisy"][cc["layer_index"]][0, cc["node_index"]])

        # (S1) At least one hidden branch crossing exists.
        if len(crossings) < 1:
            fail("branch_crossing fixture produced NO hidden Leaky-ReLU crossing.")
        print(f"[ok] branch crossing exists: {len(crossings)} hidden node(s) with "
              f"z_clean*z_noisy < 0")

        # (S2) Selected clean pre-activation is not exactly zero.
        if sel_zc == 0.0:
            fail("Selected clean pre-activation is exactly zero (ambiguous derivative).")
        print(f"[ok] selected node z_clean={sel_zc:+.6e} (nonzero, {branch_label(sel_zc)}); "
              f"z_noisy={sel_zn:+.6e} ({branch_label(sel_zn)}); "
              f"product={sel_zc*sel_zn:+.3e} < 0: {sel_zc*sel_zn < 0.0}")

        # (S5) A-vs-B difference must be clearly larger than float32-level noise.
        if err_dz_AvsB <= BRANCH_MIN_AVSB:
            fail(f"A-vs-B delta_z diff {err_dz_AvsB:.3e} not clearly larger than "
                 f"float32-level ({BRANCH_MIN_AVSB:.1e}); the crossing had no visible effect.")
        print(f"[ok] A-vs-B delta_z diff {err_dz_AvsB:.3e} >> float32-level "
              f"(threshold {BRANCH_MIN_AVSB:.1e})")

    # ------------------------------------------------------------------ #
    # Export: raw binaries + npz + manifest
    # ------------------------------------------------------------------ #
    tensors: dict[str, np.ndarray] = {"input": x_np.astype(np.float32)}
    for l in range(N_LAYERS):
        tensors[f"W{l}"] = weights[l]
        tensors[f"b{l}"] = biases[l]
        tensors[f"eps{l}"] = epsilons[l]
        tensors[f"mask{l}"] = np.ones((1, DIMS[l + 1]), dtype=np.float32)
        tensors[f"z_clean_{l}"] = res["z_clean"][l]
        tensors[f"a_clean_{l}"] = res["a_clean"][l]
        tensors[f"z_noisy_{l}"] = res["z_noisy"][l]
        tensors[f"a_noisy_{l}"] = res["a_noisy"][l]
        tensors[f"dz_original_{l}"] = res["dz_original"][l]
        tensors[f"dz_linearized_{l}"] = res["dz_linearized"][l]
    for l in range(N_LAYERS - 1):
        tensors[f"da_linearized_{l}"] = res["da_linearized"][l]
    tensors["clean_logits"] = res["clean_logits"]
    tensors["noisy_logits"] = res["noisy_logits"]
    tensors["clean_probs"] = res["clean_probs"]
    tensors["noisy_probs"] = res["noisy_probs"]
    tensors["approx_logits"] = res["approx_logits"]
    tensors["approx_probs"] = res["approx_probs"]

    # NPZ (inspection).
    npz_path = os.path.join(OUT_DIR, "golden.npz")
    np.savez(npz_path, **{k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()})

    # Raw binaries + per-file metadata.
    file_entries = []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        rel = save_bin(f"{name}.bin", a)
        file_entries.append({
            "tensor": name,
            "file": rel,
            "shape": list(a.shape),
            "count": int(a.size),
            "sha256": sha256(os.path.join(OUT_DIR, rel)),
        })

    # Determinism metadata (fixture-dependent).
    determinism = {
        "seeded": ["python.random", "numpy", "tensorflow"],
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS", ""),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        },
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__},
    }
    if fixture_name == "nominal":
        determinism["chosen_seed"] = build_meta["chosen_seed"]
        determinism["seed_search_range"] = build_meta["seed_search_range"]
        determinism["acceptance"] = build_meta["acceptance"]
    else:
        determinism["seed"] = build_meta["seed"]
        determinism["note"] = ("fixed seed; the selected hidden node's bias and epsilon "
                               "are overridden to force a Leaky-ReLU branch crossing")

    # Branch-crossing record (None for the nominal fixture).
    branch_block = None
    if fixture_name == "branch_crossing":
        cc = build_meta["crossing_construction"]
        sel_l, sel_j = cc["layer_index"], cc["node_index"]
        sel_zc = float(res["z_clean"][sel_l][0, sel_j])
        sel_zn = float(res["z_noisy"][sel_l][0, sel_j])
        sel_eps = float(epsilons[sel_l][0, sel_j])
        branch_block = {
            "crossing_layer_index": sel_l,
            "crossing_node_index": sel_j,
            "z_clean": sel_zc,
            "z_noisy": sel_zn,
            "epsilon": sel_eps,
            "clean_branch": branch_label(sel_zc),
            "noisy_branch": branch_label(sel_zn),
            "z_clean_times_z_noisy": sel_zc * sel_zn,
            "z_clean_times_z_noisy_is_negative": bool(sel_zc * sel_zn < 0.0),
            "construction": cc,
            "all_hidden_crossings": [
                {"layer": cl, "node": cj, "z_clean": zc, "z_noisy": zn,
                 "clean_branch": branch_label(zc), "noisy_branch": branch_label(zn)}
                for (cl, cj, zc, zn) in crossings
            ],
        }

    manifest = {
        "fixture": fixture_name,
        "description": (
            f"Phase-0 golden vectors ({fixture_name}): original noisy-forward ANP (A) "
            f"vs dense linearized ANP (B) for the 16->8->4->2 debug network."
        ),
        "source_of_truth": {
            "model": "models.MLP / models.DecorrelatedDense",
            "linearized_impl": "algorithms.linearized_activity_diffs",
            "activation_derivative": "algorithms._activation_derivative_elementwise",
            "note": "Golden tensors come from the repository implementations; the numpy "
                    "linearized recompute is verification-only.",
        },
        "network": {
            "dims": DIMS,
            "input_dim": INPUT_DIM,
            "hidden_sizes": HIDDEN_SIZES,
            "output_dim": OUTPUT_DIM,
            "n_layers": N_LAYERS,
            "batch_size": BATCH,
            "layers": [
                {
                    "index": l,
                    "in_dim": DIMS[l],
                    "units": DIMS[l + 1],
                    "activation": ("leaky_relu" if l < N_LAYERS - 1 else "softmax"),
                    "leaky_relu_alpha": (ALPHA if l < N_LAYERS - 1 else None),
                    "perturbed": True,
                }
                for l in range(N_LAYERS)
            ],
        },
        "conventions": {
            "dtype": DTYPE_STR,
            "endianness": "little",
            "flatten_order": "row-major (C order)",
            "weight_layout": "[in, units]; forward is z = x @ W + b",
            "noise_injection": "pre-activation: outputs_noisy = (x @ W + b) + epsilon",
            "activity_diff_definition": "delta_z = outputs_noisy - outputs_clean (pre-activation)",
            "logits_definition": "pre-softmax output-layer values",
            "softmax_definition": "probabilities (not logits)",
            "decorrelation": "disabled (plain ANP)",
            "mask": "all-ones [1, units] (dense baseline)",
        },
        "determinism": determinism,
        "self_checks": {
            "tolerance": FAIL_TOL,
            "max_abs_err_dz_original_vs_znoisy_minus_zclean": err_dz_orig,
            "max_abs_err_original_forward_reconstruction": err_orig_fwd,
            "max_abs_err_linearized_reconstruction": err_dz_lin,
            "max_abs_err_approx_logits": err_approx_logits,
            "max_abs_err_approx_probs": err_approx_probs,
        },
        "a_vs_b_boundary": {
            "compare_at": "delta_z (per layer) and output logits",
            "max_abs_diff_delta_z_original_vs_linearized": err_dz_AvsB,
            "max_abs_diff_logits_noisy_vs_approx": err_logits_AvsB,
        },
        "branch_crossing": branch_block,
        "files": {"npz": "golden.npz", "manifest": "manifest.json", "binaries": file_entries},
    }

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print("\n=== Exported tensor shapes ===")
    for e in file_entries:
        print(f"  {e['tensor']:<18} shape={tuple(e['shape'])!s:<10} -> {e['file']}")

    print("\n=== Created files ===")
    print(f"  {os.path.relpath(os.path.join(OUT_DIR, 'manifest.json'), _REPO_ROOT)}")
    print(f"  {os.path.relpath(npz_path, _REPO_ROOT)}")
    print(f"  {os.path.relpath(BIN_DIR, _REPO_ROOT)}/  ({len(file_entries)} .bin files)")
    print("\n[done] Phase-0 export complete. All self-checks passed.")


if __name__ == "__main__":
    main()
