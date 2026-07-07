# Phase-2 HLS Comparison: Direct-Original vs Exact-Reuse vs Dense-Linearized ANP

Debug network **16 -> 8 -> 4 -> 2**, batch 1, float32, Leaky ReLU (alpha = 0.2), softmax excluded.
Vitis HLS 2025.1, part `xck26-sfvc784-2LV-c`. Conservative baseline: `ap_memory` array
interfaces, `ap_ctrl_hs` control, **no** explicit unroll / array_partition / pipeline / dataflow
directives. All kernels are treated identically (same interfaces, directives, clock).

Three kernels are compared:

| ID | Kernel | Semantics | Golden reference |
|---|---|---|---|
| **A** | `anp_original_forward`   | Direct original: clean forward + **full** nonlinear noisy forward (layer-0 affine computed twice) | original |
| **B** | `anp_original_reuse_l0`  | **Exact-original** with reused first layer: `zn0 = zc0 + eps0`; noisy layers 1–2 stay full nonlinear | original |
| **C** | `anp_linearized_delta`   | Dense linearized: clean forward + first-order delta propagation | linearized |

> **A and B are algebraically identical originals** (B is an exact optimization, not an
> approximation). **C is an approximation.** All hardware numbers below are Vitis HLS
> synthesis/co-sim **estimates/measurements, not KR260 board measurements. No energy measured.**

---

## 1. Analytical operation counts (algorithmic, independent of hardware)

MAC counts from the network definition. **MAC counts do NOT equal FPGA cycles, DSPs, or LUT/FF.**

| Quantity | MACs | Derivation |
|---|---|---|
| Clean forward (common to all kernels) | **168** | 16·8 + 8·4 + 4·2 |
| A. Direct original (clean + full noisy) | **~336** | 168 + 168 |
| B. Exact reuse-L0 (clean + noisy w/ layer-0 reused) | **~208** | 168 + (168 − 128) = 168 + 40 noisy MACs; layer-0 16·8=128 matmul removed |
| C. Dense linearized (clean + delta) | **~208** | 168 + 40 delta |
| Marginal path — A noisy forward | **168** | full second forward pass |
| Marginal path — B noisy forward (layer-0 reused) | **40** | 8·4 + 4·2 (layers 1–2 only) |
| Marginal path — C delta propagation | **40** | da0·W1 (32) + da1·W2 (8) |

Note B and C have the **same marginal MAC count (40)** but reach it differently: B keeps the noisy
layers 1–2 **fully nonlinear** and only drops the redundant layer-0 matmul; C replaces the whole
noisy stack with a linearized delta recurrence. The layer-0 saving (128 MACs) is **exact and shared**
by both B and C.

---

## 2. 100 MHz results — three-way comparison (10 ns)

All three kernels **synthesized, C-simulated, and RTL-co-simulated PASS**. Full data in
[`synthesis_100mhz.csv`](synthesis_100mhz.csv) and [`cosim_results.md`](cosim_results.md).

| Metric | A. direct original | B. exact reuse-L0 | C. dense linearized |
|---|---|---|---|
| Synthesis status | PASS | PASS | PASS |
| RTL co-sim status | PASS | PASS | PASS |
| **csynth latency (cycles)** | **472** | **357** | **346** |
| **cosim latency (cycles, measured)** | **454** | **339** | **330** |
| csynth II / interval (cycles) | 473 | 358 | 347 |
| LUT | 8537 | 7250 | 6573 |
| FF | 11994 | 10007 | 9588 |
| DSP | 15 | 15 | 13 |
| BRAM | 0 | 0 | 0 |
| URAM | 0 | 0 | 0 |
| Inferred FP operator cores | 4 (2 fmul, 1 fadd, 1 faddfsub) | 4 (2 fmul, 1 fadd, 1 faddfsub) | 4 (2 fmul, 2 fadd) |
| Max abs error vs golden | 5.960e-08 | 5.960e-08 | 2.980e-08 |
| Golden reference | original | original | linearized |
| `HLS 200-885` II-violation warnings | 23 | 26 | 18 |
| Interface / control | ap_memory / ap_ctrl_hs | ap_memory / ap_ctrl_hs | ap_memory / ap_ctrl_hs |

**A and B produce the identical max error (5.960e-08) against the same original golden** →
confirms B is exact-original, not an approximation.

---

## 3. Decomposition of the 126-cycle "original vs linearized" gap (task item 8)

The previously reported gap between direct-original (A) and linearized (C) was **126 csynth cycles**
(472 − 346). Kernel B isolates how much of that was caused **only** by the redundant first-layer
affine recomputation (an exact optimization) versus the linearization approximation.

### csynth latency (static estimate)

| Step | Cycles | What it isolates |
|---|---|---|
| A → B (472 → 357) | **−115** | **Redundant first-layer affine ALONE** (exact reuse `zn0 = zc0 + eps0`) |
| B → C (357 → 346) | **−11** | **Linearization approximation** (delta vs full nonlinear noisy layers 1–2) |
| A → C total (472 → 346) | −126 | (= 115 + 11) |

**=> 115 / 126 = 91% of the original-vs-linearized latency gap was purely the duplicated
first-layer affine computation — an exact algebraic redundancy, NOT the linearization.**
Only **11 / 126 = 9%** (the B→C step) is the genuine benefit of the linearization approximation.

### cosim latency (measured RTL) — same conclusion, independent evidence

| Step | Cycles | |
|---|---|---|
| A → B (454 → 339) | **−115** | redundant affine alone |
| B → C (339 → 330) | **−9** | linearization approximation |
| A → C total (454 → 330) | −124 | (= 115 + 9) |

Measured RTL agrees: **115 / 124 = 93%** of the gap is the redundant affine; ~7% is linearization.

### Resource decomposition (100 MHz)

| Resource | A | B | C | A→B (redundant affine) | B→C (approximation) |
|---|---|---|---|---|---|
| FF  | 11994 | 10007 | 9588 | −1987 | −419 |
| LUT | 8537  | 7250  | 6573 | −1287 | −677 |
| DSP | 15    | 15    | 13   | **0** | −2 |

- The **FF saving A→B (−1987)** matches the removed noisy-layer-0 matmul loop
  (`anp_original.cpp:49`, reported at ~1887 FF / 1067 LUT) — direct hardware evidence that
  exact reuse eliminates exactly that redundant matmul.
- **DSP does not drop A→B (15 → 15):** the fp multiply/add cores are time-shared across all
  matmul loops, so removing one matmul frees *scheduling pressure* (fewer cycles, fewer pipeline
  registers) but not a whole DSP block. DSP only falls for C (13) because linearization changes
  the operator mix.

**Bottom line:** most of the apparent "linearization speed-up" over the original was actually a
fair-baseline artifact — the original kernel recomputed `x @ W0 + b0` twice. Removing that
(exact, no approximation) recovers ~91% of the gap. The linearization's *own* marginal benefit at
this network size is small (~9–11 cycles).

---

## 4. Is the duplicated first-layer `x @ W0 + b0` shared by HLS? — NO (and now fixed in B)

In `anp_original.cpp` the clean forward computes `dense(x, W0, b0, zc0)` (line 41) and the noisy
forward recomputes the **identical** `dense(x, W0, b0, zn0)` (line 49). HLS did **not** share these:
the kernel-A report contains **two independent** layer-0 matmul loop instances
(`..._Pipeline_VITIS_LOOP_28_1` ← line 41, `..._Pipeline_VITIS_LOOP_28_13` ← line 49), each
129 cycles / 1887 FF / 1067 LUT.

Kernel **B removes this at the source** (`zn0[j] = zc0[j] + eps0[j]`). Its report now contains
**five** `dense()` matmul loops (clean L0/L1/L2 + noisy L1/L2) instead of six — the noisy-layer-0
matmul is gone, confirmed by the loop→source mapping:

```
B: anp_original_reuse_l0 dense-loop instances -> anp_original_reuse.cpp:50,52,54 (clean L0/L1/L2), 60,62 (noisy L1/L2)
   (no noisy-L0 matmul; A had one at anp_original.cpp:49)
```

This is an **exact algebraic identity** (same layer-0 input for clean and noisy paths), so B matches
the original golden bit-for-bit at the tolerance (max err 5.960e-08, identical to A).

---

## 5. TIMING-FIELD CORRECTION (audit of csynth report headers)

**Prior reports mislabeled `target_period − slack` as an "achieved clock period." That is wrong
and has been removed.** The authoritative csynth "Timing" table (e.g.
`.../syn/report/anp_original_forward_csynth.rpt`) reports:

```
| Clock  | Target   | Estimated | Uncertainty |
| ap_clk | 10.00 ns | 7.114 ns  | 2.70 ns     |
```

Correct, separated fields (100 MHz, all three kernels; identical because the shared clean forward
sets the critical path):

| Field | Value | Source / meaning |
|---|---|---|
| Target clock period | **10.000 ns** | `create_clock -period 10` |
| HLS clock uncertainty | **2.70 ns** | HLS default margin (27% of target) reserved for P&R |
| HLS estimated clock period | **7.114 ns** | csynth "Estimated" — pre-implementation critical-path **estimate** |
| HLS scheduling slack | **+0.19 ns** | = Target − Uncertainty − Estimated = 10.00 − 2.70 − 7.114 |
| HLS estimated Fmax | **140.57 MHz** | `HLS 200-789` = 1 / 7.114 ns |
| **Post-implementation achieved period** | **NOT AVAILABLE** | no Vivado place-and-route, no board timing run |

250 MHz solutions (A, C): Target **4.00 ns**, Uncertainty **1.08 ns**, Estimated **2.913 ns**,
Slack **+0.01 ns**, Fmax **343.29 MHz**; post-implementation achieved period **NOT AVAILABLE**.

**Why the old label was meaningless:** `target − slack = 9.81 ns` ignores the 2.70 ns clock
uncertainty. The tool's own critical-path estimate is **7.114 ns** (⇒ 140.57 MHz), because
`slack = target − uncertainty − estimated`. So 9.81 ns corresponds to nothing the tool reports.

**No place-and-route or board timing has been measured.** Every timing number here is a
**pre-implementation HLS estimate**. A real achieved clock period would require running Vivado
synthesis + place-and-route (and a board measurement would require deployment) — neither was done.

---

## 6. Cosim-vs-csynth latency audit (measurement boundaries)

Synthesis (static) vs co-sim (measured) latency:

| Kernel | csynth (cycles) | cosim (cycles) | gap |
|---|---|---|---|
| A | 472 | 454 | 18 |
| B | 357 | 339 | 18 |
| C | 346 | 330 | 16 |

Both use the **same boundary** — one `ap_ctrl_hs` transaction, `ap_start → ap_done` (interval =
latency + 1 in *both* reports). csynth latency is the scheduler's **static worst-case estimate**;
cosim latency is XSIM's **measured** RTL cycle count (`lat.rpt`:
`$MIN_LATENCY = $MAX_LATENCY = $AVER_LATENCY`). The ~16–18 cycle gap is a **fixed scheduling-estimate
margin**, evidenced (not assumed) by:

1. cosim `min == avg == max` for every kernel ⇒ data-independent; the gap is estimate-vs-measurement,
   not input variation.
2. the absolute gap is **~constant (16–18) across three kernels of different size** (472/357/346) ⇒
   a constant control/handshake margin, not a size- or algorithm-dependent effect.
3. A and B share the **identical 18-cycle** gap despite differing by the removed matmul.

Full detail and report-file citations in [`cosim_results.md`](cosim_results.md).

---

## 7. 250 MHz characterization (4 ns) — unchanged from prior run, timing fields corrected

Kernels A and C (exact-reuse B not characterized at 250 MHz — outside the audit's scope). Both met
the HLS timing estimate. Full data in [`synthesis_250mhz.csv`](synthesis_250mhz.csv).

| Metric | A @250 | A @100 | C @250 | C @100 |
|---|---|---|---|---|
| Timing (HLS estimate) | met (+0.01 ns) | met (+0.19 ns) | met (+0.01 ns) | met (+0.19 ns) |
| HLS estimated period | 2.913 ns | 7.114 ns | 2.913 ns | 7.114 ns |
| csynth latency (cycles) | 844 | 472 | 611 | 346 |
| csynth latency (ns) | 3376 | 4720 | 2444 | 3460 |
| LUT | 10296 | 8537 | 7927 | 6573 |
| FF | 14390 | 11994 | 10959 | 9588 |
| DSP | 15 | 15 | 13 | 13 |
| FP operator cores | 4 | 4 | 4 | 4 |

**HLS did not replicate floating-point operators at 250 MHz** (core count 4→4, DSP unchanged); it
met the tighter clock by deepening pipelines (more FF/LUT, more cycles), lowering wall-clock latency
because the clock is 2.5× faster. Post-implementation achieved period: **not available** (no P&R).

---

## 8. Scope & caveats (read before citing any number)

- **HLS synthesis values are estimates, not KR260 board measurements.** No place-and-route timing
  was run; **post-implementation achieved clock period is not available.**
- **No energy / power measurement has been performed.**
- **Resource reduction is not guaranteed from MAC reduction** — e.g. DSP does not drop when the
  redundant matmul is removed (A→B: 15→15), and latency reductions are smaller than MAC ratios.
- **Kernel C (linearized) is an approximation** of the original nonlinear noisy forward.
  **Kernel B (exact reuse) is NOT an approximation** — it reproduces original ANP bit-for-bit and
  only removes a redundant recomputation.
- **Each kernel is validated against its own Python golden reference** at 1e-6 (A and B vs original
  golden; C vs linearized golden); C-sim and RTL co-sim both PASS on both fixtures.
- **Weight-update and optimizer costs are excluded** — inference-only perturbation kernels (no
  training, no DANP decorrelation, no EMA, no sparsity, no fixed-point).
- Results are for the small **16 -> 8 -> 4 -> 2** debug network only; they do not extrapolate to the
  larger 784 -> 128 -> 64 -> 10 network.
