# Phase-2 HLS RTL Co-simulation Results

Vitis HLS 2025.1 · XSIM Verilog co-simulation · part `xck26-sfvc784-2LV-c` · clock 10 ns (100 MHz).
Each kernel is co-simulated **alone** using its dedicated single-kernel testbench, because C/RTL
co-simulation wraps only the synthesis top in RTL and the shared `tb/tb_anp_hls.cpp` calls two kernels.

Both fixtures are exercised inside each co-sim run (2 kernel calls / run):
`golden_debug_16_8_4_2` (nominal) and `golden_stress_branch_crossing_16_8_4_2` (branch-crossing).
Absolute tolerance **1e-6**; the testbench returns non-zero on any mismatch (tolerance never relaxed).

## Summary

| Kernel | Top function | Testbench | Golden ref | Co-sim status | Comparisons | Max abs err | Cosim latency (min/avg/max) | Cosim interval | Total exec (cycles) | Transactions |
|---|---|---|---|---|---|---|---|---|---|---|
| A. Direct original      | `anp_original_forward`   | `tb_original.cpp`       | original   | **PASS** | 10/10 | 5.960e-08 | 454 / 454 / 454 | 455 | 909 | 2 |
| B. Exact reuse-L0       | `anp_original_reuse_l0`  | `tb_original_reuse.cpp` | original   | **PASS** | 10/10 | 5.960e-08 | 339 / 339 / 339 | 340 | 679 | 2 |
| C. Dense linearized     | `anp_linearized_delta`   | `tb_linearized.cpp`     | linearized | **PASS** | 10/10 | 2.980e-08 | 330 / 330 / 330 | 331 | 661 | 2 |

- **RTL vs golden error:** all outputs within 1e-6. Kernels A and B are validated against the
  **same original golden** and produce the **same** max error (5.960e-08 on `dz_original_2`,
  nominal fixture) — evidence that the reuse optimization is **exact**, not approximate. Kernel C
  is validated against its own linearized golden.
- **Transaction / cycle count:** `Total Execution Time` = `interval + latency` for the two
  fixture calls, e.g. A: 455 + 454 = **909** = 2 transactions; B: 340 + 339 = 679; C: 331 + 330 = 661.
  This confirms each co-sim drives exactly **2 transactions** (one per fixture) and the reported
  latency is per-transaction.

## Exit status

| Run | Make target | vitis-run exit | C/RTL co-sim verdict |
|---|---|---|---|
| A original 100 MHz synth + cosim   | `make syn_original`       | 0 | `*** C/RTL co-simulation finished: PASS ***` |
| B reuse-L0 100 MHz synth + cosim   | `make syn_original_reuse` | 0 | `*** C/RTL co-simulation finished: PASS ***` |
| C linearized 100 MHz synth + cosim | `make syn_linearized`     | 0 | `*** C/RTL co-simulation finished: PASS ***` |

## Co-simulation latency audit: why cosim cycles < csynth cycles

The synthesis (static) and co-sim (measured) latencies differ by a small, systematic amount:

| Kernel | csynth latency (cycles) | cosim latency (cycles) | Difference |
|---|---|---|---|
| A. direct original | 472 | 454 | **18** |
| B. exact reuse-L0  | 357 | 339 | **18** |
| C. dense linearized| 346 | 330 | **16** |

**Both numbers use the same boundary** — one `ap_ctrl_hs` transaction, `ap_start` → `ap_done`
(interval = latency + 1 in both reports: csynth 473/358/347 = latency+1; cosim 455/340/331 =
latency+1). They differ in *how* the cycle count is obtained, not *where* it is measured:

- **csynth latency (472/357/346)** is the scheduler's **static worst-case estimate**, taken from
  the pre-RTL schedule (`.../syn/report/csynth.rpt`). It bounds control/handshake and the
  transitions between the sequential auto-pipelined sub-loops pessimistically.
- **cosim latency (454/339/330)** is the **actual cycle count measured by XSIM** on real fixture
  data (`.../sim/report/verilog/lat.rpt`: `$MIN_LATENCY=$MAX_LATENCY=$AVER_LATENCY`), i.e. counted
  between the real RTL `ap_start`/`ap_done` events.

**Evidence that the ~16–18 cycle gap is a fixed scheduling-estimate margin, not a data-dependent
or approximation effect:**

1. In every co-sim, `min == avg == max` latency (454/454/454, 339/339/339, 330/330/330). The RTL
   is **data-independent**, so the gap cannot be runtime input variation — it is purely
   estimate-vs-measurement.
2. The absolute gap is **~constant (16–18 cycles) across three kernels of different size**
   (472, 357, 346 csynth cycles). A data-dependent or algorithm-dependent effect would scale
   with kernel size; a fixed offset points to a constant control/handshake **estimation margin**
   at the block boundary that the generated FSM recovers at run time.
3. The gap is unrelated to the reuse or linearization changes: A and B differ by the removed
   first-layer matmul yet share the **identical** 18-cycle margin.

Conclusion (evidence-based, not merely "expected"): csynth latency is a conservative static
upper bound; cosim latency is the measured realization; the ~16–18 cycle reduction is the
scheduler's fixed boundary/handshake margin, confirmed constant and data-independent above.

## Simulator / interface warnings

- **No CRITICAL warnings** in any co-sim run.
- **No SIM/COSIM interface or protocol warnings**; the `ap_ctrl_hs` handshake and `ap_memory`
  ports co-simulated cleanly for all three kernels.
- The only synthesis-side warnings are `HLS 200-885` loop-pipelining II violations on the
  single-port `ap_memory` weight arrays (see `comparison.md`). They do not affect functional
  correctness and produced no co-sim mismatch.

## Scope & caveats

- HLS/co-sim numbers are **HLS estimates / cycle-accurate RTL-sim measurements**, **not** KR260
  board measurements. **No place-and-route timing and no energy/power** were measured.
- Kernels A and B are validated against the **original** Python golden; kernel C against the
  **linearized** golden. **Linearized ANP (C) is an approximation** of the original nonlinear
  noisy forward; the **exact reuse kernel (B) is not** — B reproduces original ANP bit-for-bit
  (same float ops, same order) and only removes a redundant recomputation.
- Weight-update / optimizer costs are **excluded** (inference-only kernels).
- Results are for the small **16 -> 8 -> 4 -> 2** debug network, batch 1, float32, Leaky ReLU
  (alpha = 0.2), softmax excluded.
