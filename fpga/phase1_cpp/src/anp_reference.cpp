// Phase-1 portable CPU reference implementation.
//
// Mirrors the repository Python semantics (models.py / algorithms.py) exactly:
//   - pre-activation z = x @ W + b, weights row-major [in, units]
//   - noise epsilon added at the pre-activation of EVERY layer (output too)
//   - hidden activation leaky_relu(alpha=0.2); output activation softmax
//   - linearized: delta_a_0 = 0; delta_z_l = delta_a_{l-1} @ W_l + eps_l;
//                 delta_a_l = f'(z_clean_l) * delta_z_l (hidden only);
//                 approx_logits = z_clean_out + delta_z_out; softmax NOT linearized.
//
// float arithmetic throughout; no dynamic allocation; no STL in the math.
#include "anp_reference.hpp"

#include <cmath>   // std::exp (math function; not an STL container)

namespace anp {

real leaky_relu(real z) {
    return (z > 0.0f) ? z : (LEAKY_ALPHA * z);
}

real leaky_relu_deriv(real z) {
    // TF autodiff of leaky_relu: 1 for z > 0, alpha for z < 0. The Phase-0
    // fixture guarantees no hidden pre-activation is at/near zero, so the
    // z == 0 branch is never exercised here.
    return (z > 0.0f) ? 1.0f : LEAKY_ALPHA;
}

void softmax(const real* logits, real* probs, int n) {
    real m = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > m) m = logits[i];
    }
    real sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        real e = std::exp(logits[i] - m);
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < n; ++i) {
        probs[i] = probs[i] / sum;
    }
}

void dense(const real* x, const real* W, const real* b,
           real* z, int in_dim, int units) {
    for (int j = 0; j < units; ++j) {
        real acc = b[j];
        for (int i = 0; i < in_dim; ++i) {
            acc += x[i] * W[i * units + j];   // row-major [in, units]
        }
        z[j] = acc;
    }
}

void matvec_add_eps(const real* da, const real* W, const real* eps,
                    real* z, int in_dim, int units) {
    for (int j = 0; j < units; ++j) {
        real acc = eps[j];
        for (int i = 0; i < in_dim; ++i) {
            acc += da[i] * W[i * units + j];
        }
        z[j] = acc;
    }
}

// ---------------------------------------------------------------------------

void compute_clean(const Inputs& in, Results& r) {
    // Layer 0 (16 -> 8, leaky)
    dense(in.input, in.W0, in.b0, r.z_clean0, IN0, U0);
    for (int j = 0; j < U0; ++j) r.a_clean0[j] = leaky_relu(r.z_clean0[j]);
    // Layer 1 (8 -> 4, leaky)
    dense(r.a_clean0, in.W1, in.b1, r.z_clean1, IN1, U1);
    for (int j = 0; j < U1; ++j) r.a_clean1[j] = leaky_relu(r.z_clean1[j]);
    // Layer 2 (4 -> 2, softmax)
    dense(r.a_clean1, in.W2, in.b2, r.z_clean2, IN2, U2);
    for (int j = 0; j < U2; ++j) r.clean_logits[j] = r.z_clean2[j];
    softmax(r.z_clean2, r.clean_probs, U2);
    for (int j = 0; j < U2; ++j) r.a_clean2[j] = r.clean_probs[j];
}

void compute_original(const Inputs& in, Results& r) {
    // Noisy input is the SAME clean input (input is not perturbed).
    real tmp0[U0], tmp1[U1], tmp2[U2];

    // Layer 0
    dense(in.input, in.W0, in.b0, tmp0, IN0, U0);
    for (int j = 0; j < U0; ++j) r.z_noisy0[j] = tmp0[j] + in.eps0[j];
    for (int j = 0; j < U0; ++j) r.a_noisy0[j] = leaky_relu(r.z_noisy0[j]);
    // Layer 1
    dense(r.a_noisy0, in.W1, in.b1, tmp1, IN1, U1);
    for (int j = 0; j < U1; ++j) r.z_noisy1[j] = tmp1[j] + in.eps1[j];
    for (int j = 0; j < U1; ++j) r.a_noisy1[j] = leaky_relu(r.z_noisy1[j]);
    // Layer 2 (softmax)
    dense(r.a_noisy1, in.W2, in.b2, tmp2, IN2, U2);
    for (int j = 0; j < U2; ++j) r.z_noisy2[j] = tmp2[j] + in.eps2[j];
    for (int j = 0; j < U2; ++j) r.noisy_logits[j] = r.z_noisy2[j];
    softmax(r.z_noisy2, r.noisy_probs, U2);
    for (int j = 0; j < U2; ++j) r.a_noisy2[j] = r.noisy_probs[j];

    // Pre-activation differences (activity_diff = z_noisy - z_clean).
    for (int j = 0; j < U0; ++j) r.dz_orig0[j] = r.z_noisy0[j] - r.z_clean0[j];
    for (int j = 0; j < U1; ++j) r.dz_orig1[j] = r.z_noisy1[j] - r.z_clean1[j];
    for (int j = 0; j < U2; ++j) r.dz_orig2[j] = r.z_noisy2[j] - r.z_clean2[j];
}

void compute_linearized(const Inputs& in, Results& r) {
    // Layer 0: delta_a_{-1} = 0  =>  dz0 = eps0
    for (int j = 0; j < U0; ++j) r.dz_lin0[j] = in.eps0[j];
    for (int j = 0; j < U0; ++j) r.da_lin0[j] = leaky_relu_deriv(r.z_clean0[j]) * r.dz_lin0[j];
    // Layer 1: dz1 = da0 @ W1 + eps1
    matvec_add_eps(r.da_lin0, in.W1, in.eps1, r.dz_lin1, IN1, U1);
    for (int j = 0; j < U1; ++j) r.da_lin1[j] = leaky_relu_deriv(r.z_clean1[j]) * r.dz_lin1[j];
    // Layer 2: dz2 = da1 @ W2 + eps2; softmax NOT linearized
    matvec_add_eps(r.da_lin1, in.W2, in.eps2, r.dz_lin2, IN2, U2);
    for (int j = 0; j < U2; ++j) r.approx_logits[j] = r.z_clean2[j] + r.dz_lin2[j];
    softmax(r.approx_logits, r.approx_probs, U2);
}

void compute_reference(const Inputs& in, Results& r) {
    compute_clean(in, r);
    compute_original(in, r);
    compute_linearized(in, r);
}

} // namespace anp
