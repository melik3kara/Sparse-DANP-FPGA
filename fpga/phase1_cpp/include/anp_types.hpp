// Phase-1 portable CPU reference: fixed-size data structures.
//
// All buffers are compile-time sized (no dynamic allocation). `real` is float
// throughout, matching the Phase-0 export dtype (little-endian float32) and the
// FPGA intent (single-precision datapath).
#ifndef ANP_TYPES_HPP
#define ANP_TYPES_HPP

#include "anp_config.hpp"

namespace anp {

using real = float;

// Kernel inputs, loaded verbatim from the Phase-0 golden binaries.
// Weight layout is row-major [in, units]: W[i * units + j] = W[i][j].
struct Inputs {
    real input[INPUT_DIM];

    real W0[IN0 * U0]; real b0[U0]; real eps0[U0];
    real W1[IN1 * U1]; real b1[U1]; real eps1[U1];
    real W2[IN2 * U2]; real b2[U2]; real eps2[U2];
};

// All computed reference tensors. Names mirror the Phase-0 golden tensor names.
//   z_*  = pre-activations       a_*  = post-activations
//   *_logits = pre-softmax output values   *_probs = softmax(logits)
struct Results {
    // ----- clean forward -----
    real z_clean0[U0], a_clean0[U0];
    real z_clean1[U1], a_clean1[U1];
    real z_clean2[U2], a_clean2[U2];   // a_clean2 == clean_probs

    // ----- A. original noisy forward -----
    real z_noisy0[U0], a_noisy0[U0];
    real z_noisy1[U1], a_noisy1[U1];
    real z_noisy2[U2], a_noisy2[U2];   // a_noisy2 == noisy_probs
    real dz_orig0[U0], dz_orig1[U1], dz_orig2[U2];

    real clean_logits[OUTPUT_DIM], noisy_logits[OUTPUT_DIM];
    real clean_probs[OUTPUT_DIM],  noisy_probs[OUTPUT_DIM];

    // ----- B. dense linearized -----
    real dz_lin0[U0], dz_lin1[U1], dz_lin2[U2];
    real da_lin0[U0], da_lin1[U1];     // hidden-layer linearized delta_a only
    real approx_logits[OUTPUT_DIM], approx_probs[OUTPUT_DIM];
};

} // namespace anp

#endif // ANP_TYPES_HPP
