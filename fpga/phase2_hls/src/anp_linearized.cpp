// Phase-2 HLS kernel B: dense linearized ANP.
//
//   clean forward (common) : z_clean_l = a_clean_{l-1} @ W_l + b_l ; a = leaky_relu(z)
//   delta recurrence       : da_0 = 0
//                            dz_0 = eps_0
//                            dz_l = da_{l-1} @ W_l + eps_l
//                            da_l = leaky_relu_deriv(z_clean_l) * dz_l   (hidden only)
//   approx_logits = z_clean_out + dz_out ; logits pre-softmax (no softmax in-kernel).
//
// ap_memory array ports, ap_ctrl_hs control. Conservative baseline (no unroll/partition).
#include "anp_hls.hpp"

using namespace anp_hls;

void anp_linearized_delta(
    const float x[16],
    const float W0[16 * 8], const float b0[8], const float eps0[8],
    const float W1[8 * 4],  const float b1[4], const float eps1[4],
    const float W2[4 * 2],  const float b2[2], const float eps2[2],
    float dz0[8], float dz1[4], float dz2[2],
    float clean_logits[2], float approx_logits[2]) {

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
#pragma HLS INTERFACE mode=ap_memory port=approx_logits
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return

    // ---- Clean forward (common to both kernels) ----
    float zc0[U0], ac0[U0], zc1[U1], ac1[U1], zc2[U2];
    dense<IN0, U0>(x, W0, b0, zc0);
    for (int j = 0; j < U0; ++j) ac0[j] = leaky_relu(zc0[j]);
    dense<IN1, U1>(ac0, W1, b1, zc1);
    for (int j = 0; j < U1; ++j) ac1[j] = leaky_relu(zc1[j]);
    dense<IN2, U2>(ac1, W2, b2, zc2);

    // ---- Linearized delta propagation ----
    float d0[U0], da0[U0], d1[U1], da1[U1], d2[U2];
    for (int j = 0; j < U0; ++j) d0[j] = eps0[j];                       // dz_0 = eps_0
    for (int j = 0; j < U0; ++j) da0[j] = leaky_relu_deriv(zc0[j]) * d0[j];
    matvec_add_eps<IN1, U1>(da0, W1, eps1, d1);                         // dz_1
    for (int j = 0; j < U1; ++j) da1[j] = leaky_relu_deriv(zc1[j]) * d1[j];
    matvec_add_eps<IN2, U2>(da1, W2, eps2, d2);                         // dz_2

    // ---- Outputs ----
    for (int j = 0; j < U0; ++j) dz0[j] = d0[j];
    for (int j = 0; j < U1; ++j) dz1[j] = d1[j];
    for (int j = 0; j < U2; ++j) dz2[j] = d2[j];
    for (int j = 0; j < U2; ++j) { clean_logits[j] = zc2[j]; approx_logits[j] = zc2[j] + d2[j]; }
}
