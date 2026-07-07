# Phase 1 — Portable C++ scalar reference (16 → 8 → 4 → 2)

Plain C++17 CPU reference for the two ANP paths, validated against the Phase‑0
golden binaries in `../phase0/golden_debug_16_8_4_2/bin`. **No HLS pragmas, no
Vitis, no external libraries.** This is the correctness baseline that a later
Vitis HLS kernel (Phase 2) must reproduce.

## What it computes

Same semantics as the repository Python (`models.py` / `algorithms.py`):

- Pre-activation `z = x @ W + b`, weights row-major `[in, units]`.
- Noise `epsilon` added at the **pre-activation** of **every** layer (output too).
- Hidden activation `leaky_relu(alpha=0.2)`; output activation `softmax`
  (numerically stable: subtracts the max logit).
- **A. Original noisy-forward ANP:** full nonlinear noisy forward;
  `dz_original_l = z_noisy_l − z_clean_l`.
- **B. Dense linearized ANP:** `da_0=0`; `dz_l = da_{l-1} @ W_l + eps_l`;
  `da_l = f'(z_clean_l) · dz_l` (hidden only); `approx_logits = z_clean_out + dz_out`.
  The softmax is **not** linearized.

`logits` = pre-softmax output values; `probs` = `softmax(logits)`.

## Files

```
fpga/phase1_cpp/
├── include/
│   ├── anp_config.hpp      # compile-time dims, alpha=0.2, tolerance=1e-6
│   ├── anp_types.hpp       # fixed-size Inputs / Results (float, no dynamic alloc)
│   └── anp_reference.hpp   # reference API
├── src/
│   └── anp_reference.cpp   # dense, leaky_relu(+deriv), stable softmax, both paths
├── test/
│   └── test_phase1.cpp     # loads golden .bin, compares, prints PASS/FAIL, sets exit code
├── Makefile
└── README.md
```

The math in `anp_reference.cpp` uses only fixed-size arrays and `float`; the only
non-stdlib header is `<cmath>` (`std::exp`). The testbench uses host file I/O
(`<cstdio>`, `std::string`) to read the golden binaries — allowed for the host TB.

## Build

```
cd fpga/phase1_cpp
make
# or explicitly:
g++ -O2 -std=c++17 -Wall -Wextra -Iinclude src/anp_reference.cpp test/test_phase1.cpp -o test_phase1
```

## Run

```
cd fpga/phase1_cpp
./test_phase1 ../phase0/golden_debug_16_8_4_2/bin
# or:
make run
```

The golden directory can be overridden: `./test_phase1 <path/to/bin>` or
`make run GOLDEN=<path>`. Default is `../phase0/golden_debug_16_8_4_2/bin`
(i.e. run from `fpga/phase1_cpp`). The program returns **non-zero** if any golden
comparison exceeds the tolerance or any file fails to load.

## Result (this machine)

Compiler: `g++ -O2 -std=c++17`. Golden vectors: TF 2.17.1 / NumPy 1.26.4, seed 1.
All **26** comparisons PASS at absolute tolerance **1e-6**; worst error **1.788e-7**
(single-precision dot-product ordering vs TensorFlow — well within tolerance, so
the tolerance was **not** relaxed). Exit code **0**.

A-vs-B diagnostic (C++): `max|dz_original − dz_linearized| = 6.706e-8`,
`max|noisy_logits − approx_logits| = 5.960e-8`.

> **Interpretation.** These A-vs-B differences are tiny **only because this
> fixture's small perturbations never cross a Leaky-ReLU branch boundary** (no
> hidden pre-activation changes sign under noise). This validates *implementation
> correctness*, **not** general equivalence of original and linearized ANP.

## Proposed (NOT yet implemented) stress fixture

To exercise the genuine approximation gap, add a second Phase‑0 fixture where at
least one hidden pre-activation sits close enough to zero that the perturbation
flips its Leaky-ReLU branch (`sign(z_clean) ≠ sign(z_clean + contribution)`):

- Reuse `phase0_export_golden.py` with an alternate config
  `golden_stress_16_8_4_2/`, changing the acceptance rule from "avoid near-zero"
  to "**require** ≥1 hidden node with `|z_clean| < c` and `|epsilon-driven Δz| > |z_clean|`
  for that node", so the noisy forward and the linearized derivative (frozen at
  `f'(z_clean)`) diverge.
- Optionally raise `noise_std` (e.g. 0.1–0.3) so `Δz` reliably crosses zero.
- Export the **same** tensor set. The C++ TB here runs unchanged against it; the
  A-vs-B diagnostic should then show a **non-negligible** `dz_original` vs
  `dz_linearized` gap at the branch-crossing node — the intended demonstration.

This is a proposal only; no stress fixture or new export has been created yet.
