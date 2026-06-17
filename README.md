# DANP — Decorrelated Activity-based Node Perturbation
This project is based on the official DANP implementation by Sander Dalm and extends it with sparse perturbation policies and FPGA-oriented analysis. 

## Quick start

```bash
# CIFAR-10, 3-layer MLP, 3 seeds
python main.py --dataset cifar10 --algorithm danp \
    --hidden_sizes 1024 1024 1024 --epochs 50 \
    --batch_size 1000 --lr 1e-4 --noise_std 0.001 \
    --num_noise_iters 1 --num_seeds 3 \
    --write_results_dir results/cifar10_fc

python plot.py --results_folder results/cifar10_fc
```

## Sparse DANP Experiments

### Motivation

Node Perturbation perturbs every hidden node on every forward pass.  On
neuromorphic or FPGA hardware this is expensive: each perturbed node requires
independent noise injection and a separate weight update.  Restricting
perturbation to a sparse subset of k < H nodes per layer reduces the compute
and energy cost by a factor of k/H, while (ideally) preserving learning quality.

This codebase implements and compares six node-selection policies under the
`--sparse` flag.  All policies produce a binary mask of shape [1, H] per layer.
The mask is applied to layer noise before the noisy forward pass; unselected
nodes carry zero noise AND zero gradient, so both perturbation and update are
sparse.

Enable sparse mode with:

```bash
python main.py --algorithm danp --sparse --sparse_fraction 0.5 \
    --sparse_policy <policy> ...
```

### Implemented policies

| Policy | Selection criterion | Notes |
|---|---|---|
| `random` | Uniformly random k-subset each step | Unbiased gradient estimator; dropout-like regularisation |
| `scheduled` | Deterministic round-robin, shifts by k each step | Guarantees full coverage over H/k steps; FPGA-friendly |
| `activation_threshold` | Top-k by mean \|clean activation\| (pre-noise) | Single-pass; biases toward high-activation nodes |
| `activity_diff_topk` | Top-k by mean \|Δactivation\| (provisional pass) | Two-pass; ranks by response magnitude, ignores loss sign |
| `activity_loss_topk` | Top-k by mean \|Δactivation × Δloss\| (provisional pass) | Two-pass; ranks by ANP numerator — loss-correlated response |
| `gradient_aligned_topk` | Top-k by mean \|Δactivation × Δloss / ‖Δact‖²\| (provisional pass) | Two-pass; ranks by full ANP effective-error magnitude g_i |

The three two-pass policies run a provisional full-noise forward pass to
estimate selection scores, then zero out noise for unselected nodes before the
actual training forward pass.

### Current findings

Across MNIST and CIFAR-10, **random sparse DANP at 50 % active nodes** matches
or outperforms all adaptive policies tested so far.  The key theoretical
explanation is ANP's own normalisation:

    g_i  ∝  activity_diff_i × performance_diff / ‖activity_diff‖²

Selecting nodes by \|activity_diff\| (or even by \|activity_diff × perf_diff\|)
tends to pick nodes that also inflate the denominator ‖activity_diff‖², so
the normalised update is not necessarily larger for selected nodes.  Random
selection is an unbiased estimator of the full gradient and additionally
provides a dropout-like regularisation effect.

The `--analyze_gradient_dist` flag measures how concentrated or diffuse the
per-node signal g_i actually is during training:

```bash
python main.py --algorithm danp --epochs 5 --num_seeds 3 \
    --analyze_gradient_dist --exp_name cifar10_danp_graddist
```

Results are saved to `<exp_dir>/gradient_dist.json`.  Key statistics:

- **CV** (coefficient of variation): near 0 → diffuse; > 1 → concentrated
- **top_10pct**: fraction of total \|g\| energy in the top 10 % of nodes;
  > 0.5 → concentrated, ≈ 0.1 → diffuse
- **Gini**: 0 = equal signal across all nodes; ≈ 1 = one node dominates

If the signal is empirically diffuse, random sparse perturbation is near-optimal
and adaptive policies offer no theoretical advantage — a concrete, falsifiable
conclusion relevant to FPGA sparse-perturbation design.

---

## All example commands

```bash
# CIFAR 3 layers
python main.py --dataset cifar10 --algorithm np --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm anp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm inp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python main.py --dataset cifar10 --algorithm dnp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm danp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc
python main.py --dataset cifar10 --algorithm dinp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python main.py --dataset cifar10 --algorithm bp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-4 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python main.py --dataset cifar10 --algorithm dbp --hidden_sizes 1024 1024 1024 --epochs 50 --batch_size 1000 --lr 1e-3 --noise_std 0.001 --num_noise_iters 1 --num_seeds 3 --write_results_dir results/cifar10_fc

python plot.py --results_folder results/cifar10_fc

# MNIST 50 iters
python main.py --dataset mnist --algorithm anp --hidden_sizes 20 20 --epochs 200 --batch_size 1 --lr 1e-5 --noise_std 0.001 --num_noise_iters 50 --num_seeds 1 --write_results_dir results/mnist --gpu 5
```