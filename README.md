# Sparse ANP / DANP for FPGA-Oriented Online Learning

**Research question:** Can sparse, hardware-friendly node perturbation preserve
near-dense ANP/DANP learning performance while reducing perturbation-related
training cost components for FPGA-oriented online learning systems?
---

## Table of Contents

1. [Background: ANP in brief](#background)
2. [Repository layout](#layout)
3. [How to run](#how-to-run)
4. [Experiments](#experiments)
   - [E1 Sparse vs full (DANP, h=1024)](#e1)
   - [E2 Adaptive top-k policies](#e2)
   - [E3 Gradient distribution](#e3)
   - [E4 Width sweep](#e4)
   - [E5 Random vs scheduled (ANP, h=[64,64])](#e5)
   - [E6 Layer allocation](#e6)
   - [E7 Gaussian vs Rademacher](#e7)
   - [E8 Budget sweep + coverage floor (Fashion-MNIST)](#e8)
   - [E9 Antithetic / resample / single](#e9)
   - [E10 EMA normalizer: amortizing the ANP global norm](#e10)
   - [E11 Linearized ANP: replacing the nonlinear noisy forward](#e11)
   - [E12 Lazy DANP decorrelation: amortizing R updates](#e12)
5. [Key findings](#findings)

---

## 1. Background: ANP in brief <a name="background"></a>

Activity-based Node Perturbation (ANP) [Jabri & Flower, IEEE Trans. Neural Netw.,
1992] trains a neural network without backpropagation. Each step:

1. **Clean forward pass** → activations a_l and loss L_clean.
2. **Noisy forward pass** — add small noise v_l to each layer's pre-activations →
   perturbed activations and loss L_noisy.
3. **Weight update:**

   ΔW_l = N · δL · (δa_l / ‖δa‖²) · x_{l-1}^T

   where δL = L_clean − L_noisy  and  δa_l = a_l(noisy) − a_l(clean).

**Why ANP, not plain NP?** Classic Node Perturbation (NP) uses the injected noise v_l
directly as the update direction. ANP uses the *measured activation difference* δa_l
instead. This matters because noise injected into layer l−1 also shifts activations in
layer l; the activation difference captures that propagated effect while the injected
noise alone does not. ANP can provide a more informative update direction than raw NP in multilayer
networks, because δa_l includes the propagated effect of upstream perturbations.
The update does not rely directly on the raw injected noise vectors; it uses
clean/noisy activations and layer inputs.

**Cost vs backprop:** 2 forward passes (clean + noisy), no backward pass.

**DANP** (Decorrelated ANP) adds a per-step whitening update R ← R − η_dec · C̃R
applied to each layer's inputs. It stabilises training on harder tasks such as
CIFAR-10 where plain ANP collapses (see [E1](#e1)).

**Noise standard deviation:** σ = 0.01 (σ² = 1e-4) in all experiments. Two
distributions were tested:
- **Gaussian:** ε_i ~ N(0, σ²) per active node.
- **Rademacher:** ε_i = ±σ with equal probability — same zero mean and per-node
  variance as Gaussian, requires only one random bit per node in hardware.

**Sparse perturbation:** a binary mask activates only fraction α of nodes per layer.
Inactive nodes receive zero perturbation and do not contribute to the sparse
perturbation update for that layer. The active-node cost proxy therefore scales
approximately with α.

This does not mean that total training FLOPs scale with α: the clean and noisy
forward matrix multiplications remain dense in the current implementation. Therefore,
α should be interpreted as a perturbation/update-side proxy, not a full hardware
runtime or energy measurement.

Two mask policies:
- **random** — fresh random subset each step.
- **scheduled** — deterministic round-robin counter that shifts by k each step,
  guaranteeing every node participates in exactly one window of H/k steps. Requires
  only a counter in hardware (no sorting, no ranking).

---

## 2. Repository layout <a name="layout"></a>

```
.
├── README.md
├── main.py                      # entry point; all experiments run through this
├── src/
│   ├── algorithms.py        # perturbation_gradients(), train_step()
│   ├── models.py            # MLP, DecorrelatedDense
│   ├── sparse.py            # allocate_fractions(), compute_sparse_masks()
│   ├── coverage.py          # per-layer coverage stats
│   ├── analysis.py          # gradient distribution stats (CV, Gini, Top-10%)
│   ├── utils.py             # dataset loaders, evaluation, I/O helpers
│   ├── scripts/
│   │   ├── sweep_budget.py  # grid sweep: fraction × policy × algorithm × seed
│   │   └── sweep_lr.py      # LR grid sweep
│   └── tests/
│       └── test_sal_equivalence.py  # SAL-ANP losslessness check (36 cases)
├── reports/
│   ├── progress_report_v4.pdf
│   ├── progress_report_v5.pdf
│   └── PROGRESS.md
└── results/                     # experiment outputs (not tracked in git)
```

Each `results/<exp>/` directory contains:
- `run_summary.csv` — one row per seed; columns include `algorithm`, `hidden_sizes`,
  `sparse_fraction`, `sparse_policy`, `noise_distribution`, `noise_sampling`,
  `noisy_passes`, `best_test_acc`, `rel_cost_final`.
- `peak_test_acc.txt` — max test accuracy across seeds and epochs.
- `<exp_name>.json` — full argument config saved at run time.
- Per-metric `_mean.txt`, `_std.txt`, `_traj.txt` trajectory files.

---

## 3. How to run <a name="how-to-run"></a>

### Dependencies

No `requirements.txt` is present. The code imports:

```
tensorflow >= 2.10   (tested with 2.15)
numpy
```

### General invocation

```bash
python main.py \
  --dataset <mnist|fashion_mnist|cifar10> \
  --algorithm <anp|danp> \
  --hidden_sizes 64 64 \
  --epochs 20 \
  --lr 0.001 \
  --noise_std 0.01 \
  --sparse \
  --sparse_fraction 0.25 \
  --sparse_policy scheduled \
  --noise_distribution rademacher \
  --noise_sampling single \
  --num_seeds 3 \
  --seed 42 \
  --write_results_dir results \
  --exp_name my_experiment
```

### Key flags

| Flag | Meaning | Values used |
|---|---|---|
| `--dataset` | Training dataset | `mnist`, `fashion_mnist`, `cifar10` |
| `--algorithm` | Training algorithm | `anp` (no decorr.), `danp` (with decorr.) |
| `--hidden_sizes` | Hidden layer widths (space-separated) | `64 64`, `256 256 256`, `1024 1024 1024` |
| `--epochs` | Training epochs | 20 (most runs), 5 (early DANP runs) |
| `--lr` | Adam learning rate | 0.001 |
| `--decor_lr` | Decorrelation matrix step size (DANP only) | 0.001 |
| `--noise_std` | Perturbation amplitude σ | 0.01 |
| `--batch_size` | Minibatch size | 500 (all ANP [64,64] experiments), 1000 (DANP [1024³] experiments — from saved config) |
| `--sparse` | Enable sparse perturbation | flag (absent = dense) |
| `--sparse_fraction` | Active fraction α = k/H | 0.0156 – 0.5 |
| `--sparse_policy` | Node selection policy | `random`, `scheduled`, `activity_loss_topk`, `activity_diff_topk`, `update_magnitude_topk` |
| `--layer_allocation` | Budget split across layers | `uniform`, `front_loaded`, `back_loaded`, `middle_loaded` |
| `--noise_distribution` | Noise type | `gaussian`, `rademacher` |
| `--noise_sampling` | One or two noisy passes | `single`, `antithetic`, `resample` |
| `--norm_mode` | ANP normalizer mode | `exact` (default), `ema`, `none` |
| `--norm_beta` | EMA decay for `--norm_mode ema` | 0.99 |
| `--norm_update_every` | Recompute ‖δa‖² every K steps | 1, 20, 100 |
| `--delta_mode` | How to compute activity differences | `noisy` (default), `linearized`, `compare` |
| `--decor_update_every` | DANP decorrelation R update frequency | 1, 5, 20, 100 |
| `--probe_layer` | Ablation: perturb only one layer | integer index or `all` |
| `--num_seeds` | Number of seeds run from `--seed` | 3 |

---

## 4. Experiments <a name="experiments"></a>

### E1 — Sparse vs full perturbation (DANP, h=1024) <a name="e1"></a>

**Motivation:** Establish that sparse perturbation can match full perturbation at a
fraction of the cost, using a wide network where the hypothesis is non-trivial.

**Architecture:** DANP, `[1024, 1024, 1024]` (three hidden layers; 3×1024 + 10 = 3082
perturbable nodes). Adam lr=0.001, decor_lr=0.001, batch_size=1000, noise_std=0.01.

```bash
# Full DANP, MNIST, 20 epochs
python main.py --dataset mnist --algorithm danp \
  --hidden_sizes 1024 1024 1024 --epochs 20 \
  --num_seeds 3 --seed 0 \
  --write_results_dir results --exp_name mnist_danp_full_20

# Scheduled 25%, MNIST, 20 epochs
python main.py --dataset mnist --algorithm danp \
  --hidden_sizes 1024 1024 1024 --epochs 20 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --num_seeds 3 --seed 0 \
  --write_results_dir results --exp_name mnist_danp_scheduled_25_20
```

**MNIST results** (peak test accuracy, no per-seed CSV; n=3 seeds):

| Policy | α | Active / Total | Rel. Cost | Peak Acc. |
|---|---|---|---|---|
| Full DANP | — | 3082 / 3082 | 1.000 | 0.9642 |
| Random | 25% | 770 / 3082 | 0.250 | 0.9558 |
| Scheduled | 25% | 770 / 3082 | 0.250 | 0.9566 |
| ActivityLoss topk | 25% | 770 / 3082 | 0.250 | 0.9314 * |
| ActivityDiff topk | 25% | 770 / 3082 | 0.250 | 0.9313 * |

\* Run for only 5 epochs — not comparable to the 20-epoch rows; see [E2](#e2).

At 25% active nodes (75% reduction in the active-node perturbation/update proxy) DANP drops < 0.008 accuracy on MNIST.

**CIFAR-10 results** (DANP, 5 epochs, peak, no per-seed CSV; n=3):

```bash
python main.py --dataset cifar10 --algorithm danp \
  --hidden_sizes 1024 1024 1024 --epochs 5 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --num_seeds 3 --seed 0 \
  --write_results_dir results --exp_name cifar10_danp_scheduled_25
```

| Policy | α | Rel. Cost | Peak Acc. |
|---|---|---|---|
| Full DANP | — | 1.000 | 0.4094 |
| Scheduled | 50% | 0.500 | 0.4122 |
| Scheduled | 25% | 0.250 | 0.4102 |
| Random | 50% | 0.500 | 0.4080 |
| Random | 25% | 0.250 | 0.4070 |
| ActivityLoss topk | 50% | 0.500 | 0.4109 |
| ActivityLoss topk | 25% | 0.250 | 0.4055 |
| GradientAligned topk | 50% | 0.500 | 0.4117 |
| GradientAligned topk | 25% | 0.250 | 0.4076 |

All sparse policies are within ±0.004 of full DANP at 5 epochs on CIFAR-10.

> **Plain ANP on CIFAR-10** (no decorrelation): an earlier dense pilot on a smaller
> MLP was unstable and sometimes collapsed to chance-level accuracy (~0.10). All
> CIFAR-10 conclusions in this report are therefore based on DANP, not plain ANP.

---

### E2 — Adaptive top-k policies vs random/scheduled <a name="e2"></a>

**Motivation:** We hypothesised that selecting the nodes with the highest activity
change (ActivityDiff), loss contribution (ActivityLoss), or gradient alignment
(GradientAligned) would outperform random selection.

**Epoch mismatch on MNIST:** The random/scheduled 25% runs used 20 epochs; adaptive
runs used 5 epochs. A direct comparison at 25% budget on MNIST is not available.

**Fair comparisons that exist:**

*MNIST, α=50%, 5 epochs* (from `peak_test_acc.txt`, no per-seed CSV; n=3 seeds):

| Policy | α | Epochs | Peak Acc. |
|---|---|---|---|
| Random | 50% | 5 | 0.9407 |
| ActivityLoss topk | 50% | 5 | 0.9398 |
| ActivityDiff topk | 50% | 5 | 0.9377 |

At matched epoch count and budget, ActivityLoss and ActivityDiff both fall below
random selection.

*CIFAR-10 (all policies, 5 epochs):* see the CIFAR table in [E1](#e1). Scheduled and
random match or exceed all adaptive policies across both budget levels.

In no tested configuration does an adaptive policy consistently outperform random or
scheduled. See [E3](#e3) for the mechanistic explanation.

---

### E3 — Gradient distribution analysis <a name="e3"></a>

**Motivation:** Understand why adaptive selection fails. If the gradient signal were
concentrated (high Gini), top-k selection should help. If diffuse, coverage is the
bottleneck.

**Method:** `--analyze_gradient_dist`, DANP [1024,1024,1024], 5 epochs, full
perturbation, MNIST and CIFAR-10, 3 seeds.

```bash
python main.py --dataset mnist --algorithm danp \
  --hidden_sizes 1024 1024 1024 --epochs 5 \
  --analyze_gradient_dist \
  --num_seeds 3 --seed 0 \
  --write_results_dir results --exp_name mnist_danp_graddist
```

**Results** (from progress report logs — mean across 3 seeds; per-epoch CSV not
available for this format):

*MNIST (full DANP, 5 epochs):*

| Epoch | CV | Top 10% | Gini |
|---|---|---|---|
| 1 | 0.238 | 0.135 | 0.134 |
| 2 | 0.253 | 0.138 | 0.142 |
| 3 | 0.263 | 0.141 | 0.148 |
| 4 | 0.272 | 0.143 | 0.152 |
| 5 | 0.281 | 0.144 | 0.156 |

*CIFAR-10 (full DANP, 5 epochs):*

| Epoch | CV | Top 10% | Gini |
|---|---|---|---|
| 1 | 0.228 | 0.132 | 0.129 |
| 3 | 0.238 | 0.134 | 0.134 |
| 5 | 0.245 | 0.135 | 0.138 |

All three concentration statistics are low. The top 10% of nodes carry only ~13–15%
of gradient energy (a perfectly uniform signal would give 10%). The signal is
effectively diffuse throughout training on both datasets. This is why simple coverage
policies (random, scheduled) match adaptive top-k ones: there is no concentrated
signal to exploit.

---

### E4 — Width sweep <a name="e4"></a>

**Motivation:** Check whether the accuracy–sparsity tradeoff holds across network
sizes, not just at h=1024.

**Method:** DANP, 3-hidden-layer networks of widths [64,64,64], [256,256,256],
[1024,1024,1024]. Scheduled 25% sparse vs full DANP, MNIST, 20 epochs, 3 seeds.

```bash
python main.py --dataset mnist --algorithm danp \
  --hidden_sizes 64 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --num_seeds 3 --seed 0 \
  --write_results_dir results --exp_name mnist_danp_sched25_h64
```

**Results** (peak test accuracy, no per-seed CSV; n=3):

| Architecture | Total nodes | Rel. Cost | Full Acc. | Sparse Acc. | Drop |
|---|---|---|---|---|---|
| `[64, 64, 64]` | 202 | 0.248 | 0.9391 | 0.9342 | −0.0049 |
| `[256, 256, 256]` | 778 | 0.249 | 0.9530 | 0.9480 | −0.0050 |
| `[1024, 1024, 1024]` | 3082 | 0.250 | 0.9642 | 0.9566 | −0.0076 |

The 0.005–0.008 accuracy drop at 25% sparsity is consistent across all tested widths.

---

### E5 — Random vs scheduled (ANP, h=[64,64], MNIST, Gaussian) <a name="e5"></a>

**Motivation:** Switch to a smaller, more FPGA-realistic network. Verify that the
random/scheduled comparison holds at 138 total nodes and characterise the Gaussian
budget curve to baseline for Rademacher ([E7](#e7)).

**Architecture:** ANP (no decorrelation), `[64, 64]` (two hidden layers + 10-class
output = 138 total perturbable nodes). Gaussian noise, 20 epochs, batch_size=500.

```bash
python main.py --dataset mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --noise_distribution gaussian \
  --num_seeds 3 --seed 42 \
  --write_results_dir results --exp_name budget_anp_scheduled_0.25
```

**Dense baseline (α=1.0, Gaussian, n=3):** 0.9291 ± 0.0017

**Budget sweep (n=3, mean ± std, best test accuracy):**

| Policy | α | Active / 138 | Rel. Cost | Mean ± Std |
|---|---|---|---|---|
| Dense | 1.0 | 138 | 1.000 | 0.9291 ± 0.0017 |
| Scheduled | 0.500 | 69 | 0.500 | 0.9299 ± 0.0019 |
| Scheduled | 0.250 | 34 | 0.246 | 0.9262 ± 0.0008 |
| Scheduled | 0.125 | 17 | 0.125 | 0.9266 ± 0.0028 |
| Scheduled | 0.0625 | 9 | 0.065 | 0.9257 ± 0.0026 |
| Random | 0.500 | 69 | 0.500 | 0.9273 ± 0.0023 |
| Random | 0.250 | 34 | 0.246 | 0.9263 ± 0.0027 |
| Random | 0.125 | 17 | 0.125 | 0.9239 ± 0.0024 |
| Random | 0.0625 | 9 | 0.065 | 0.9221 ± 0.0021 |

All differences between scheduled and random at the same α are within 2× the combined
standard deviations. Scheduled consistently matches or slightly exceeds random, without
any statistical significance claim at n=3.

---

### E6 — Layer allocation <a name="e6"></a>

**Motivation:** When the sparse budget is fixed at α=25%, should it be distributed
uniformly across layers, or biased toward early (broader reach) or late (closer to
loss) layers?

**Method:** ANP [64,64], MNIST, 20 epochs, α=0.25, scheduled policy, Gaussian noise.

```bash
python main.py --dataset mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --layer_allocation back_loaded \
  --num_seeds 3 --seed 42 \
  --write_results_dir results --exp_name layeralloc_anp_back_loaded
```

**Results (n=3, mean ± std):**

| Allocation | Mean ± Std |
|---|---|
| uniform | 0.9290 ± 0.0030 |
| front_loaded | 0.9278 ± 0.0016 |
| middle_loaded | 0.9202 ± 0.0020 |
| back_loaded | 0.9155 ± 0.0018 |

Uniform matches front_loaded (differences < 1 std). Back_loaded is clearly worst:
concentrating budget in the 10-node output layer is inefficient. Uniform is the safe
default.

---

### E7 — Gaussian vs Rademacher noise <a name="e7"></a>

**Motivation:** Rademacher noise (±σ, one random bit) is cheaper to generate on FPGA
than multi-bit Gaussian. We tested whether the single-bit approximation hurts accuracy.

**Method:** ANP [64,64], MNIST + Fashion-MNIST, scheduled policy, 20 epochs, 3 seeds.

```bash
python main.py --dataset mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.125 --sparse_policy scheduled \
  --noise_distribution rademacher \
  --num_seeds 3 --seed 42 \
  --write_results_dir results --exp_name rademacher_scheduled_0.125
```

**MNIST dense baselines (α=1.0, n=3):**

| Noise | Mean ± Std |
|---|---|
| Gaussian | 0.9291 ± 0.0017 |
| Rademacher | 0.9299 ± 0.0022 |

**MNIST Rademacher scheduled budget sweep (n=3):**

| α | Rel. Cost | Mean ± Std | Gaussian equivalent |
|---|---|---|---|
| 0.500 | 0.500 | 0.9280 ± 0.0013 | 0.9299 ± 0.0019 |
| 0.250 | 0.246 | 0.9277 ± 0.0028 | 0.9262 ± 0.0008 |
| 0.125 | 0.125 | 0.9288 ± 0.0005 | 0.9266 ± 0.0028 |
| 0.0625 | 0.065 | 0.9258 ± 0.0023 | 0.9257 ± 0.0026 |

**Fashion-MNIST dense pilot (α=1.0, 20 epochs, n=3):**

| Noise | Mean ± Std |
|---|---|
| Gaussian | 0.8209 ± 0.0062 |
| Rademacher | 0.8191 ± 0.0036 |

All Rademacher results are within one standard deviation of the corresponding Gaussian
results. Rademacher is the preferred choice going forward.

---

### E8 — Budget sweep and coverage floor (Fashion-MNIST, extreme sparsity) <a name="e8"></a>

**Motivation:** MNIST saturates quickly (>0.92 with only 9 active nodes), so we
switched to Fashion-MNIST to probe the low-budget regime more clearly. We pushed α
down to 1.56% (≈2–3 active nodes per layer) to find the floor below which accuracy
degrades noticeably.

**Architecture:** ANP [64,64] (138 nodes), Rademacher, 20 epochs, scheduled and
random policies, lr=0.001, batch_size=500.

```bash
# Scheduled, α = 1.56%
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.015625 --sparse_policy scheduled \
  --noise_distribution rademacher \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_rademacher_budget_extreme \
  --exp_name fashion_rad_sched_0.015625

# Random, α = 1.56%
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.015625 --sparse_policy random \
  --noise_distribution rademacher \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_rademacher_random_policy \
  --exp_name fashion_rad_random_0.015625
```

**Scheduled results (n=3, mean ± std):**

| α | Active nodes * | Rel. Cost | Mean ± Std |
|---|---|---|---|
| Dense (1.0) | 138 | 1.000 | 0.8191 ± 0.0036 |
| 0.5000 | 69 | 0.500 | 0.8206 ± 0.0044 |
| 0.2500 | 34 | 0.246 | 0.8248 ± 0.0035 |
| 0.1250 | 17 | 0.123 | 0.8147 ± 0.0079 |
| 0.0625 | 9 | 0.065 | 0.8114 ± 0.0052 |
| 0.0313 | 5 | 0.036 | 0.8043 ± 0.0017 |
| 0.0156 | 3 | 0.022 | 0.7863 ± 0.0075 |

\* Active-node counts and relative costs are taken from the saved run summaries
(`rel_cost_final` column). Small deviations from nominal α arise from integer rounding
per layer.

**Random results (n=3, mean ± std):**

| α | Rel. Cost | Mean ± Std |
|---|---|---|
| 0.5000 | 0.500 | 0.8228 ± 0.0033 |
| 0.2500 | 0.246 | 0.8161 ± 0.0014 |
| 0.1250 | 0.123 | 0.8137 ± 0.0040 |
| 0.0625 | 0.065 | 0.8140 ± 0.0049 |
| 0.0313 | 0.036 | 0.8102 ± 0.0030 |
| 0.0156 | 0.022 | 0.7903 ± 0.0045 |

**Key observations:**
- The knee in accuracy vs budget is around α = 0.125–0.25 (17–34 active nodes). Below
  α = 0.0625 the decline accelerates.
- At α = 0.0156 (≈3 active nodes total), std rises to 0.0075 (scheduled) and 0.0045
  (random) — seed-level variance is high because individual seeds can collapse when
  per-layer active count drops to 1.
- At α = 0.25, scheduled (0.8248) > random (0.8161), a gap of 0.0087. This exceeds
  both individual standard deviations but does not persist at other budgets — treat as
  an outlier, not a reliable trend.

**Probe-layer ablation** (MNIST, ANP [64,64], α=0.25 scheduled, 20 epochs, n=3):

| Layer | Units | Mean ± Std |
|---|---|---|
| 0 (first hidden) | 64 | 0.9324 ± 0.0019 |
| 1 (second hidden) | 64 | 0.8149 ± 0.0078 |
| 2 (output) | 10 | 0.7427 ± 0.0139 |

Perturbing only layer 0 nearly matches full perturbation; perturbing only the output
layer is much worse. Consistent with the gradient distribution finding: first-layer
noise propagates through the whole network and is most informative. (But uniform
allocation outperforms front-loaded in [E6](#e6) — the output layer still needs some
budget.)

---

### E9 — Antithetic / resample / single <a name="e9"></a>

**Motivation:** At very low budgets the gradient estimate has high variance. Antithetic
sampling averages the +v and −v gradient estimates (same mask), which cancels the
even-order term in the Taylor expansion of δL and in principle reduces bias. Resample
(two independent draws on the same mask) is the matched-cost control: it tests whether
any two-pass averaging helps at all.

**Modes:**
- `single` — one noisy pass (default, all prior experiments).
- `antithetic` — two passes: +v then −v on the same mask and same nodes; average both
  gradient estimates. Cost: 2× noisy passes.
- `resample` — two passes: independent v1, v2 on the same mask; average. Cost: 2×.

**Method:** ANP [64,64], Fashion-MNIST, Rademacher, scheduled, 20 epochs,
α ∈ {0.0156, 0.0313, 0.0625}, 3 seeds per mode.

```bash
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.015625 --sparse_policy scheduled \
  --noise_distribution rademacher --noise_sampling antithetic \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_antithetic_anp \
  --exp_name fashion_anp_antithetic_0.015625
```

**Results (n=3, mean ± std, best test accuracy):**

| Mode | α = 0.0156 | α = 0.0313 | α = 0.0625 |
|---|---|---|---|
| single | 0.7935 ± 0.0038 | 0.8068 ± 0.0030 | 0.8076 ± 0.0021 |
| antithetic | 0.7945 ± 0.0070 | 0.8046 ± 0.0031 | 0.8124 ± 0.0037 |
| resample | 0.7987 ± 0.0048 | 0.8132 ± 0.0032 | 0.8165 ± 0.0025 |

**Interpretation:**
- **Antithetic vs single:** differences are within one std at all budgets. With σ=0.01
  the curvature term that antithetic cancels is already negligible — no gain.
- **Resample vs single:** a consistent positive trend of +0.005–0.009. However at n=3
  all differences are within ~1–1.5 combined standard deviations — inconclusive.
- **Cost:** both modes double the per-step noisy-pass count. Even if resample's trend
  is real, the gain does not justify the extra pass for FPGA deployment.
- **Recommendation:** `--noise_sampling single` (default).

---

### E10 — EMA Normalizer: Amortizing the ANP Global Norm <a name="e10"></a>

**Motivation:** The ANP weight update divides the activity-difference error by the
network-wide activity norm ‖δa‖²:

    ΔW_l ∝ δL · δa_l / ‖δa‖² · x_{l-1}^T

Computing ‖δa‖² exactly requires a global reduction over all active-layer outputs every
training step. On FPGA, this means reading all activations into a reduction tree and
then performing a per-sample division — an operation that cannot be easily pipelined
when the global norm is needed before the weight update can proceed.

The hypothesis is that ‖δa‖² changes slowly relative to weight updates, so an
exponential moving average (EMA) can replace the exact per-step value without
measurable accuracy loss. Refreshing the EMA every K steps amortises the global
reduction cost by K and, in hardware, allows the per-step division to be replaced by
multiplication with a stored scalar reciprocal.

**This experiment does not reduce active-node count.** It is orthogonal to the sparse
perturbation experiments above. The active-node cost (α) is fixed; the change is solely
in how the normalizer ‖δa‖² is computed.

**New CLI flags:**

```
--norm_mode {exact, ema, none}   default: exact
--norm_beta FLOAT                EMA decay; default: 0.99
--norm_update_every INT          recompute norm every K steps; default: 1
--norm_eps FLOAT                 denominator epsilon; default: 1e-8
```

**Method:** ANP [64,64], Fashion-MNIST, Rademacher, scheduled, 20 epochs,
α ∈ {0.0625, 0.125, 0.25}, 3 seeds per condition.

```bash
# exact (baseline, K=1)
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.0625 --sparse_policy scheduled \
  --noise_distribution rademacher \
  --norm_mode exact \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_norm_ablation \
  --exp_name fashion_norm_exact_0.0625

# EMA, K=20
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.125 --sparse_policy scheduled \
  --noise_distribution rademacher \
  --norm_mode ema --norm_beta 0.99 --norm_update_every 20 \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_norm_ablation \
  --exp_name fashion_norm_ema_k20_0.125

# EMA, K=100
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --noise_distribution rademacher \
  --norm_mode ema --norm_beta 0.99 --norm_update_every 100 \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_norm_ablation \
  --exp_name fashion_norm_ema_k100_0.25
```

**Results (n=3, mean ± std best test accuracy, Fashion-MNIST):**

| α | norm mode | K | mean best acc | std | mean final acc | min final acc | active-node cost |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0625 | exact | 1 | 0.8161 | 0.0012 | 0.7848 | 0.7536 | 0.065 |
| 0.0625 | ema | 1 | 0.8084 | 0.0010 | 0.8057 | 0.8013 | 0.065 |
| 0.0625 | ema | 20 | 0.8038 | 0.0062 | 0.7074 | 0.6108 | 0.065 |
| 0.0625 | ema | 100 | 0.8085 | 0.0050 | 0.8080 | 0.8027 | 0.065 |
| 0.125 | exact | 1 | 0.8126 | 0.0047 | 0.8061 | 0.7991 | 0.123 |
| 0.125 | ema | 1 | 0.8166 | 0.0055 | 0.8064 | 0.8001 | 0.123 |
| 0.125 | ema | 20 | 0.8148 | 0.0057 | 0.8086 | 0.7996 | 0.123 |
| 0.125 | ema | 100 | 0.8222 | 0.0019 | 0.8059 | 0.7968 | 0.123 |
| 0.25 | exact | 1 | 0.8163 | 0.0041 | 0.8077 | 0.7980 | 0.246 |
| 0.25 | ema | 1 | 0.8203 | 0.0026 | 0.8086 | 0.7920 | 0.246 |
| 0.25 | ema | 20 | 0.8201 | 0.0042 | 0.8164 | 0.8113 | 0.246 |
| 0.25 | ema | 100 | 0.8194 | 0.0043 | 0.8184 | 0.8155 | 0.246 |

Because `active-node cost` reports only the sparse perturbation/update proxy, the
normalizer saving is not reflected in that column. EMA normalization targets a separate
cost component: the global ‖δa‖² reduction and division.

**Interpretation:** EMA normalization preserves ANP learning performance while reducing
the frequency of the exact global normalization computation. Across Fashion-MNIST sparse
budgets, EMA-normalized ANP matches exact normalization in best accuracy within
seed-level variation. At α=0.125 and α=0.25, EMA with K=100 performs comparably to
exact normalization while updating the global norm estimate only once every 100 steps.
At α=0.0625, exact normalization gives the highest mean best accuracy, but EMA K=100
gives better final-training stability. EMA K=20 is unstable at α=0.0625 (mean final acc
0.7074, min 0.6108), so the effect is not monotonic and should not be overclaimed.

**Hardware note:** Exact normalization requires a per-step global reduction over all
‖δa‖² values and a division at every weight-update step. EMA normalization replaces
this with a stored scalar estimate. In hardware, the reciprocal of the EMA value can be
precomputed or updated only when the EMA is refreshed, so the per-step division can be
replaced by multiplication with a stored reciprocal. This is an FPGA-oriented
approximation to reduce normalization overhead. It does not reduce the cost of the main
forward passes or weight-update matrix multiplications, which remain unchanged.

**Limitation:** We did not sweep β. All EMA runs used β=0.99, so the EMA result is a refresh-frequency ablation rather than a beta-sensitivity study. Since β=0.99 is a slow-moving average, especially when combined with large K, future work should test β ∈ {0.9, 0.95, 0.99}.
---

### E11 — Linearized ANP: Replacing the Nonlinear Noisy Forward <a name="e11"></a>

**Motivation:** Standard ANP computes a clean forward pass and then a full nonlinear
noisy forward pass to measure activity differences. In the small-noise regime, the
noisy activity difference can be approximated by first-order perturbation propagation:

    δa_l ≈ f'(z_l) ⊙ (W_l δa_{l-1} + ε_l)

where z_l comes from the clean pass, f'(z_l) is the clean activation derivative mask,
and δa_0 = 0. This replaces the full nonlinear noisy activation recomputation with
tangent/delta propagation.

**Important scope:** Linearized ANP does not remove the clean forward pass. It also does
not necessarily remove all noisy-pass matrix multiplication: dense delta propagation can
still require dense matmuls. The expected hardware benefit comes from avoiding nonlinear
noisy activation recomputation, reducing the need to store full noisy activations, and
enabling sparse delta propagation in future implementations.

**New CLI flags:**

```
--delta_mode {noisy, linearized, compare}   default: noisy
```

- `noisy`: original ANP behavior — full clean forward + full nonlinear noisy forward.
- `linearized`: replaces the noisy forward with first-order delta propagation. No
  `forward_noisy()` call is made; δz_l is computed from the clean-pass derivative masks
  and propagated through the network.
- `compare`: diagnostic mode — runs both paths on the same batch/noise/mask, trains on
  the noisy path, and logs approximation-quality metrics per batch
  (`rel_delta_z_error`, `rel_delta_logits_error`, `rel_delta_L_error`).

**Method:** ANP [64,64], Fashion-MNIST, Rademacher noise, scheduled sparse masks, exact
normalizer, 20 epochs, batch_size=500, lr=0.001, σ=0.01, n=3 seeds.

```bash
# noisy (baseline)
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.125 --sparse_policy scheduled \
  --noise_distribution rademacher --noise_sampling single \
  --delta_mode noisy \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_linearized_pilot \
  --exp_name fashion_linearized_noisy_0.125

# linearized
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 20 \
  --sparse --sparse_fraction 0.125 --sparse_policy scheduled \
  --noise_distribution rademacher --noise_sampling single \
  --delta_mode linearized \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/fashion_linearized_pilot \
  --exp_name fashion_linearized_lin_0.125

# compare (diagnostic)
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 5 \
  --sparse --sparse_fraction 0.125 --sparse_policy scheduled \
  --noise_distribution rademacher --noise_sampling single \
  --delta_mode compare \
  --num_seeds 1 --seed 42 \
  --write_results_dir results/fashion_linearized_pilot \
  --exp_name fashion_linearized_compare_0.125
```

Compare mode is diagnostic-only. It is not used for the main accuracy table; it is used
to measure how closely the linearized delta path approximates the full noisy forward on
the same batch/noise/mask.

**Results (n=3, Fashion-MNIST):**

| α | delta mode | mean best acc | std best | mean final acc | min final acc | active-node cost | nonlinear noisy activation evals saved / batch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0625 | noisy | 0.8094 | 0.0015 | 0.7952 | 0.7764 | 0.065 | 0 |
| 0.0625 | linearized | 0.8112 | 0.0045 | 0.7966 | 0.7859 | 0.065 | 64000 |
| 0.125 | noisy | 0.8174 | 0.0095 | 0.7886 | 0.7740 | 0.123 | 0 |
| 0.125 | linearized | 0.8147 | 0.0042 | 0.7842 | 0.7455 | 0.123 | 64000 |
| 0.25 | noisy | 0.8210 | 0.0029 | 0.8176 | 0.8109 | 0.246 | 0 |
| 0.25 | linearized | 0.8192 | 0.0013 | 0.8127 | 0.8031 | 0.246 | 64000 |

**Interpretation:** Linearized ANP matches the original noisy-forward ANP within
seed-level variation across all tested sparse budgets. The mean-best accuracy difference
is small: +0.0018 at α=0.0625, −0.0027 at α=0.125, and −0.0018 at α=0.25. This supports
the small-noise hypothesis that, at σ=0.01, the nonlinear noisy forward can be
approximated by first-order delta propagation without a meaningful accuracy loss.

However, final-epoch stability is not uniformly better. At α=0.125, linearized ANP has
a lower minimum final accuracy (0.7455 vs 0.7740), so robustness should not be
overclaimed.

**Hardware note:** The saved count `64000` is a proxy for hidden-layer nonlinear
activation evaluations avoided per batch: batch_size × (64 + 64) = 500 × 128. It is not
a full FLOP or energy measurement. The current implementation still performs dense delta
propagation. Therefore, this experiment should be interpreted as evidence that nonlinear
noisy activation recomputation and full noisy activation storage may be avoidable, not
as a measured total runtime reduction.

#### Compare-mode diagnostic

To directly verify the small-noise approximation, we ran `--delta_mode compare` across
the same Fashion-MNIST sparse budgets. Compare mode runs both the full nonlinear noisy
forward path and the first-order linearized delta path on the same batch, same noise
draw, and same sparse mask, then logs the approximation error between them. Training
still uses the noisy path; compare mode is diagnostic-only and is not used for the
accuracy results above.

**Method:** Fashion-MNIST, ANP [64,64], Rademacher noise, scheduled sparse masks, exact
normalizer, σ=0.01, batch_size=500, lr=0.001, 5 epochs, n=3 seeds,
α ∈ {0.0625, 0.125, 0.25}.

```bash
python main.py --dataset fashion_mnist --algorithm anp \
  --hidden_sizes 64 64 --epochs 5 \
  --batch_size 500 --lr 0.001 --noise_std 0.01 \
  --sparse --sparse_fraction 0.125 --sparse_policy scheduled \
  --layer_allocation uniform \
  --noise_distribution rademacher --noise_sampling single \
  --norm_mode exact \
  --delta_mode compare \
  --seed 42 \
  --write_results_dir results/fashion_linearized_compare \
  --exp_name fashion_compare_0.125_seed42
```

**Results (n=3 seeds, last-epoch mean per condition):**

| α | active-node cost | mean rel_δz | mean rel_δlogits | mean abs_δL | mean rel_δL |
|---:|---:|---:|---:|---:|---:|
| 0.0625 | 0.065 | 0.00161 | 0.00425 | 1.9e-06 | 0.062 |
| 0.125 | 0.123 | 0.00235 | 0.00661 | 3.4e-06 | 0.096 |
| 0.25 | 0.246 | 0.00296 | 0.00836 | 4.6e-06 | 0.113 |

Metric definitions:
- `rel_δz`: relative error between noisy and linearized activity differences,
  ‖δz_noisy − δz_lin‖ / ‖δz_noisy‖, averaged over the batch and epoch.
- `rel_δlogits`: same, for the output logit differences.
- `abs_δL`: absolute difference |δL_noisy − δL_lin|, averaged over the batch.
- `rel_δL`: |δL_noisy − δL_lin| / |δL_noisy|.

**Interpretation:** The activity and logit errors are small: `rel_δz` stays below 0.30%
and `rel_δlogits` below 0.84% across all tested budgets. Both increase mildly with α,
which is expected since more active nodes means a larger aggregate first-order
approximation error. The `rel_δL` values (6–11%) look moderate but must be read
alongside `abs_δL`: the absolute δL gap is in the range 1.9–4.6 × 10⁻⁶, because δL
itself is very small at σ=0.01. Relative δL error inflates when the denominator is tiny;
the absolute error is the more informative quantity here.

Together with the noisy vs. linearized accuracy comparison above, these diagnostics
support the small-noise explanation: at σ=0.01, the full nonlinear noisy forward and
the first-order linearized delta path remain close on the same inputs, and this
closeness is consistent with the observed accuracy match. This does not validate the
approximation at larger σ, on other datasets, or with wider networks.

---

### E12 — Lazy DANP Decorrelation: Amortizing R Updates <a name="e12"></a>

**Motivation:** DANP adds decorrelation matrices R on top of ANP. There are two
separate operations:

1. Applying the current R matrices every step.
2. Updating R using the decorrelation update.

This experiment only makes the R *update* lazy. R is still applied every training
step. The weight update frequency is unchanged.

**New CLI flag:**

```
--decor_update_every K   default: 1
```

K=1 is the original DANP behavior — R is updated every step. K>1 updates R only
every K global training steps; R is still applied (unchanged) on every step in
between.

**Method:** CIFAR-10, DANP [256,256,256], scheduled sparse α=0.25, Rademacher
noise, 5 epochs, batch_size=500, lr=0.001, decor_lr=0.001, σ=0.01, n=3 seeds.

```bash
# K=5
python main.py --dataset cifar10 --algorithm danp \
  --hidden_sizes 256 256 256 --epochs 5 \
  --batch_size 500 --lr 0.001 --decor_lr 0.001 \
  --sparse --sparse_fraction 0.25 --sparse_policy scheduled \
  --noise_distribution rademacher \
  --decor_update_every 5 \
  --num_seeds 3 --seed 42 \
  --write_results_dir results/cifar10_lazy_danp_pilot \
  --exp_name cifar10_danp_lazy_k5
```

**Results (n=3, CIFAR-10):**

| K | mean best acc | std best | mean final acc | min final acc | decor update fraction | mean updates | total steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3902 | 0.0060 | 0.3902 | 0.3843 | 1.000 | 500.0 | 500.0 |
| 5 | 0.3868 | 0.0110 | 0.3868 | 0.3752 | 0.200 | 100.0 | 500.0 |
| 20 | 0.3609 | 0.0023 | 0.3609 | 0.3585 | 0.050 | 25.0 | 500.0 |
| 100 | 0.3186 | 0.0086 | 0.3186 | 0.3087 | 0.010 | 5.0 | 500.0 |

**Interpretation:** K=5 matches K=1 within seed-level variation while reducing the
decorrelation-update frequency from 100% to 20%. K=20 and K=100 clearly degrade
CIFAR-10 accuracy, so decorrelation can be amortized but not made too infrequent.

**Hardware/cost note:** `decor_update_fraction` is a proxy for DANP-specific
decorrelation-update frequency. It is not a measured FPGA energy or runtime
reduction. Applying R and the main ANP/DANP weight updates still occur every step.

---

## 5. Key findings <a name="findings"></a>

1. **Sparse perturbation preserves accuracy.** At α=25% (75% reduction in the
   active-node perturbation/update proxy), DANP drops < 0.008 on MNIST across all
   tested widths. On Fashion-MNIST (harder), 12.5% active nodes (87% reduction in the
   active-node perturbation/update proxy) costs about 0.004–0.006 relative to α=50%.

2. **Adaptive top-k selection adds nothing.** ActivityLoss, ActivityDiff, and
   GradientAligned policies match or underperform random and scheduled at matched
   epochs and budgets. Root cause: the gradient signal is broadly distributed (Gini ≈
   0.13–0.16), so no node is consistently more valuable than others.

3. **Coverage is the real lever.** Both random and scheduled ensure every node
   participates periodically. Scheduled mask generation is more hardware-friendly than
   random or top-k policies because it can be implemented with a counter and avoids
   sorting/ranking. In the tested experiments, scheduled is broadly comparable to
   random in accuracy. Uniform layer allocation is the safest default among the tested
   allocations.

4. **Rademacher noise has no measurable accuracy cost.** Replacing Gaussian with
   Rademacher (one random bit per node) causes no measurable accuracy drop at any
   tested budget on MNIST or Fashion-MNIST. It is the preferred choice for FPGA.

5. **Two-pass variance reduction is not worth the doubled cost.** Antithetic shows no
   gain. Resample shows a small positive trend but it is inconclusive at n=3 and
   doubles noisy-pass count. Use `--noise_sampling single`.

6. **Exact per-step ANP normalization is not always necessary.** On Fashion-MNIST, EMA
   normalization with K=100 preserved near-exact best accuracy at α=0.125 and α=0.25,
   suggesting that the global ‖δa‖² reduction can be amortized for FPGA-oriented online
   learning. The effect is not universal: at α=0.0625, EMA K=20 was unstable. Further
   validation across seeds and datasets is needed before claiming robustness.

7. **Linearized ANP preserves learning in the small-noise regime.** On Fashion-MNIST,
   replacing the full nonlinear noisy forward with first-order delta propagation matched
   noisy ANP within seed-level variation across α ∈ {0.0625, 0.125, 0.25}. This suggests
   that ANP's noisy pass can be approximated using clean-pass derivative masks at σ=0.01,
   although dense delta matmuls still remain and FPGA runtime savings are not yet measured.

8. **Lazy DANP decorrelation can reduce R-update frequency.** On CIFAR-10, K=5
   preserved performance within seed-level variation while reducing R updates to
   20%, but K≥20 degraded accuracy.

