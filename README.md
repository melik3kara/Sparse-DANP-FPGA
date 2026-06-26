# Sparse ANP / DANP for FPGA-Oriented Online Learning

**Research question:** Can sparse node perturbation reach comparable accuracy to full
node perturbation while significantly reducing training energy and compute cost for
FPGA-based online learning systems?
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
5. [Key findings](#findings)
6. [Limitations and next steps](#limitations)
7. [Missing / unverified data](#missing)

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
noise alone does not. ANP therefore produces a better gradient estimate without storing
the noise vectors — only the two sets of activations are needed.

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
Inactive nodes receive zero noise and no weight update. Cost ∝ α.

Two mask policies:
- **random** — fresh random subset each step.
- **scheduled** — deterministic round-robin counter that shifts by k each step,
  guaranteeing every node participates in exactly one window of H/k steps. Requires
  only a counter in hardware (no sorting, no ranking).

---

## 2. Repository layout <a name="layout"></a>

```
.
├── main.py              # entry point; all experiments run through this
├── algorithms.py        # perturbation_gradients(), train_step()
├── models.py            # MLP, DecorrelatedDense (reset_noise, forward_noisy)
├── sparse.py            # allocate_fractions(), compute_sparse_masks()
├── coverage.py          # per-layer coverage stats (mask / effective / norms)
├── analysis.py          # gradient distribution stats (CV, Gini, Top-10%)
├── utils.py             # dataset loaders, evaluation, I/O helpers
├── sweep_budget.py      # grid sweep helper (fractions × policies × algorithms)
├── sweep_lr.py          # learning-rate grid helper
├── plot.py / plot_sweep.py  # plotting utilities
└── results/
    ├── dense_anp_baseline/          # ANP [64,64] Gaussian dense, MNIST
    ├── dense_rademacher_baseline/   # ANP [64,64] Rademacher dense, MNIST
    ├── budget_sweep_anp/            # ANP [64,64] Gaussian random+scheduled sweep, MNIST
    ├── rademacher_budget_sweep/     # ANP [64,64] Rademacher scheduled sweep, MNIST
    ├── layer_allocation/            # ANP [64,64] uniform/front/back/middle, MNIST
    ├── probe_layer/                 # ANP [64,64] single-layer probe, MNIST
    ├── fashion_dense/               # ANP [64,64] Gaussian+Rademacher dense, Fashion-MNIST
    ├── fashion_rademacher_budget/   # ANP [64,64] Rademacher scheduled, Fashion (α≥0.0625)
    ├── fashion_rademacher_budget_extreme/  # same, α=0.015625 and 0.03125
    ├── fashion_rademacher_random_policy/   # ANP [64,64] Rademacher random, Fashion
    ├── fashion_antithetic_anp/      # ANP [64,64] single/antithetic/resample, Fashion
    ├── cifar10_dense_pilot/         # ANP [64,64] Gaussian+Rademacher dense, CIFAR-10
    ├── mnist_danp_full_20/          # DANP [1024,1024,1024] full, MNIST  †
    ├── mnist_danp_random_25_20/     # DANP [1024,1024,1024] random 25%, MNIST  †
    ├── mnist_danp_scheduled_25_20/  # DANP [1024,1024,1024] scheduled 25%, MNIST  †
    ├── mnist_danp_actloss_25/       # DANP activity-loss topk 25%, MNIST (5ep)  †
    ├── mnist_danp_sparse_actdiff_25/# DANP activity-diff topk 25%, MNIST (5ep)  †
    ├── cifar10_danp_full/           # DANP [1024,1024,1024] full, CIFAR-10  †
    ├── cifar10_danp_scheduled_25/   # DANP scheduled 25%, CIFAR-10  †
    ├── cifar10_danp_gradalign_25/   # DANP gradient-aligned topk 25%, CIFAR-10  †
    ├── mnist_danp_full_h64/         # DANP [64,64,64] full, MNIST (width sweep)  †
    ├── mnist_danp_sched25_h64/      # DANP [64,64,64] scheduled 25%, MNIST  †
    ├── mnist_danp_full_h256/        # DANP [256,256,256] full, MNIST  †
    ├── mnist_danp_sched25_h256/     # DANP [256,256,256] scheduled 25%, MNIST  †
    └── ...
```

† Directories marked with † predate the `run_summary.csv` format. Numbers from these
are taken from `peak_test_acc.txt` (peak over all seeds and epochs; no per-seed
breakdown). All other experiments have `run_summary.csv` with one row per seed.

Each `results/<exp>/` directory contains:
- `run_summary.csv` — one row per seed (where available); columns include
  `algorithm`, `hidden_sizes`, `sparse_fraction`, `sparse_policy`,
  `noise_distribution`, `noise_sampling`, `noisy_passes`, `best_test_acc`,
  `rel_cost_final`.
- `peak_test_acc.txt` — max test accuracy across seeds and epochs (all experiments).
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
| `--sparse_policy` | Node selection policy | `random`, `scheduled`, `activity_loss_topk`, `activity_diff_topk`, `gradient_aligned_topk` |
| `--layer_allocation` | Budget split across layers | `uniform`, `front_loaded`, `back_loaded`, `middle_loaded` |
| `--noise_distribution` | Noise type | `gaussian`, `rademacher` |
| `--noise_sampling` | One or two noisy passes | `single`, `antithetic`, `resample` |
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

At 25% active nodes (75% cost reduction) DANP drops < 0.008 accuracy on MNIST.

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

## 5. Key findings <a name="findings"></a>

1. **Sparse perturbation preserves accuracy.** At α=25% (75% cost reduction), DANP
   drops < 0.008 on MNIST across all tested widths. On Fashion-MNIST (harder), 12.5%
   active nodes (87% cost reduction) costs about 0.004–0.006 relative to α=50%.

2. **Adaptive top-k selection adds nothing.** ActivityLoss, ActivityDiff, and
   GradientAligned policies match or underperform random and scheduled at matched
   epochs and budgets. Root cause: the gradient signal is broadly distributed (Gini ≈
   0.13–0.16), so no node is consistently more valuable than others.

3. **Coverage is the real lever.** Both random and scheduled ensure every node
   participates periodically. Scheduled (a counter, not a LFSR + comparator) is
   strictly cheaper in hardware and achieves the same or slightly better accuracy.
   Uniform layer allocation is the safest default among the tested allocations.

4. **Rademacher noise has no measurable accuracy cost.** Replacing Gaussian with
   Rademacher (one random bit per node) causes no measurable accuracy drop at any
   tested budget on MNIST or Fashion-MNIST. It is the preferred choice for FPGA.

5. **Two-pass variance reduction is not worth the doubled cost.** Antithetic shows no
   gain. Resample shows a small positive trend but it is inconclusive at n=3 and
   doubles noisy-pass count. Use `--noise_sampling single`.

---

## 6. Limitations and next steps <a name="limitations"></a>

**Limitations:**

- **n=3 seeds** throughout. Differences smaller than ≈2× the reported std should be
  treated as trends, not established results. No significance tests are reported.
- **MNIST saturates.** Above α ≈ 0.06 (9 nodes), MNIST accuracy changes by < 0.004
  — the budget sweep is more informative on Fashion-MNIST.
- **Plain ANP on CIFAR-10 collapses.** All CIFAR-10 experiments require DANP. An
  earlier dense pilot of plain ANP (smaller MLP) reached chance-level accuracy (~0.10);
  DANP's decorrelation step is required for CIFAR-10.
- **No FPGA resource measurement.** All cost figures are relative node counts (α).
  Actual energy, LUT utilisation, and throughput on target hardware have not been
  measured.
- **Gradient distribution stats from earlier logs.** The CV/Gini/Top-10% values (E3)
  are from a pre-CSV logging format; no per-seed breakdown is available.

**Next steps:**

1. **FPGA cost model on Kria.** Translate α → energy and resource savings; measure
   throughput at different sparsity levels.
2. **Reduce computation inside the algorithm.** Candidates: skip ‖δa‖² normalisation
   periodically (constant normaliser); reduce decorrelation update frequency; quantise
   the weight update.
3. **Confirm resample trend** with more seeds (n ≥ 5) before committing to it.
4. **Harder benchmarks** (CIFAR-100, or an edge-vision task) once the FPGA pipeline is
   in place.

---

## 7. Missing / unverified data <a name="missing"></a>

| Item | Status |
|---|---|
| Per-seed accuracy for all DANP h=1024 experiments | No `run_summary.csv`; only `peak_test_acc.txt` (peak across 3 seeds) |
| Per-epoch CV, Gini, Top-10% for gradient distribution | Pre-CSV format; means from progress report logs |
| MNIST random 50% at 20 epochs, DANP h=1024 | Experiment not run at 20 epochs |
| MNIST scheduled 50% at 20 epochs, DANP h=1024 | Experiment not run |
| ANP [64,64] CIFAR-10 with sparse perturbation | Not run; only dense pilot exists |
| `noise_std` and `noise_distribution` for layer_allocation and probe_layer | Not in CSV (pre-feature addition); inferred as `gaussian` / `0.01` from code defaults |
