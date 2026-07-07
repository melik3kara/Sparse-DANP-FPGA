// Phase-2 HLS kernel C: exact-original ANP with reused first-layer preactivation.
//
// Same original noisy-forward ANP semantics as anp_original_forward, but the noisy
// layer-0 preactivation is obtained from the clean layer-0 preactivation instead of
// recomputing x @ W0 + b0 a second time:
//
//     clean : zc0 = x @ W0 + b0
//     noisy : zn0 = zc0 + eps0        <-- EXACT algebraic identity (same layer-0 input)
//
// This is NOT linearization and NOT an approximation: because clean and noisy paths
// share the identical network input at layer 0, x @ W0 + b0 is common to both, so
// zn0 = (x @ W0 + b0) + eps0 = zc0 + eps0 bit-for-bit (same float ops, same order).
//
// Later noisy layers (1, 2) remain FULL nonlinear noisy-forward layers, identical to
// anp_original_forward -- they are NOT reused and NOT linearized.
//
// ap_memory array ports, ap_ctrl_hs control. Conservative baseline (no unroll/partition).
#include "anp_hls.hpp"

using namespace anp_hls;

void anp_original_reuse_l0(
    const float x[16],
    const float W0[16 * 8], const float b0[8], const float eps0[8],
    const float W1[8 * 4],  const float b1[4], const float eps1[4],
    const float W2[4 * 2],  const float b2[2], const float eps2[2],
    float dz0[8], float dz1[4], float dz2[2],
    float clean_logits[2], float noisy_logits[2]) {

    // ---- Interface: on-chip (ap_memory) arrays; block-level ap_ctrl_hs ----
#pragma HLS INTERFACE mode=ap_memory port=x
#pragma HLS INTERFACE mode=ap_memory port=W0
#pragma HLS INTERFACE mode=ap_memory port=b0
#pragma HLS INTERFACE mode=ap_memory port=eps0
#pragma HLS INTERFACE mode=ap_memory port=W1
#pragma HLS INTERFACE mode=ap_memory port=b1
#pragma HLS INTERFACE mode=ap_memory port=eps1
#pragma HLS INTERFACE mode=ap_memory port=W2
#pragma HLS INTERFACE mode=ap_memory port=b2
#pragma HLS INTERFACE mode=ap_memory port=eps2
#pragma HLS INTERFACE mode=ap_memory port=dz0
#pragma HLS INTERFACE mode=ap_memory port=dz1
#pragma HLS INTERFACE mode=ap_memory port=dz2
#pragma HLS INTERFACE mode=ap_memory port=clean_logits
#pragma HLS INTERFACE mode=ap_memory port=noisy_logits
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return

    // ---- Clean forward ----
    float zc0[U0], ac0[U0], zc1[U1], ac1[U1], zc2[U2];
    dense<IN0, U0>(x, W0, b0, zc0);
    for (int j = 0; j < U0; ++j) ac0[j] = leaky_relu(zc0[j]);
    dense<IN1, U1>(ac0, W1, b1, zc1);
    for (int j = 0; j < U1; ++j) ac1[j] = leaky_relu(zc1[j]);
    dense<IN2, U2>(ac1, W2, b2, zc2);

    // ---- Noisy forward: reuse clean layer-0 preactivation (zn0 = zc0 + eps0) ----
    // Layer 0 affine (x @ W0 + b0) is NOT recomputed. Layers 1,2 are full noisy layers.
    float zn0[U0], an0[U0], zn1[U1], an1[U1], zn2[U2];
    for (int j = 0; j < U0; ++j) { zn0[j] = zc0[j] + eps0[j]; an0[j] = leaky_relu(zn0[j]); }
    dense<IN1, U1>(an0, W1, b1, zn1);
    for (int j = 0; j < U1; ++j) { zn1[j] += eps1[j]; an1[j] = leaky_relu(zn1[j]); }
    dense<IN2, U2>(an1, W2, b2, zn2);
    for (int j = 0; j < U2; ++j) zn2[j] += eps2[j];

    // ---- Outputs: per-layer pre-activation differences and logits (same as kernel A) ----
    for (int j = 0; j < U0; ++j) dz0[j] = zn0[j] - zc0[j];
    for (int j = 0; j < U1; ++j) dz1[j] = zn1[j] - zc1[j];
    for (int j = 0; j < U2; ++j) dz2[j] = zn2[j] - zc2[j];
    for (int j = 0; j < U2; ++j) { clean_logits[j] = zc2[j]; noisy_logits[j] = zn2[j]; }
}
