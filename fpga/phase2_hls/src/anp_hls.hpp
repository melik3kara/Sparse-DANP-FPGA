// Phase-2 Vitis HLS: shared synthesizable declarations for the two ANP kernels.
//
// Debug network 16 -> 8 -> 4 -> 2, batch 1, float. Weights row-major [in, units]
// (W[i*units + j]). Hidden activation leaky_relu(alpha=0.2); NO softmax in-kernel.
// Both kernels compute the common clean forward internally.
//
// Conservative baseline: no unroll / no array_partition / no pipeline pragmas.
#ifndef ANP_HLS_HPP
#define ANP_HLS_HPP

namespace anp_hls {

// ----- Compile-time dimensions -----
constexpr int IN0 = 16, U0 = 8;   // layer 0: 16 -> 8  (leaky_relu)
constexpr int IN1 = 8,  U1 = 4;   // layer 1: 8  -> 4  (leaky_relu)
constexpr int IN2 = 4,  U2 = 2;   // layer 2: 4  -> 2  (linear; softmax done off-kernel)

constexpr float LEAKY_ALPHA = 0.2f;

// ----- Elementwise activation (synthesizable) -----
inline float leaky_relu(float z)       { return (z > 0.0f) ? z : LEAKY_ALPHA * z; }
inline float leaky_relu_deriv(float z) { return (z > 0.0f) ? 1.0f : LEAKY_ALPHA; }

// z[j] = b[j] + sum_i x[i] * W[i*UNITS + j]   (row-major [IN, UNITS])
template <int IN, int UNITS>
inline void dense(const float x[IN], const float W[IN * UNITS],
                  const float b[UNITS], float z[UNITS]) {
    for (int j = 0; j < UNITS; ++j) {
        float acc = b[j];
        for (int i = 0; i < IN; ++i) acc += x[i] * W[i * UNITS + j];
        z[j] = acc;
    }
}

// z[j] = eps[j] + sum_i da[i] * W[i*UNITS + j]   (linearized propagation; bias cancels)
template <int IN, int UNITS>
inline void matvec_add_eps(const float da[IN], const float W[IN * UNITS],
                           const float eps[UNITS], float z[UNITS]) {
    for (int j = 0; j < UNITS; ++j) {
        float acc = eps[j];
        for (int i = 0; i < IN; ++i) acc += da[i] * W[i * UNITS + j];
        z[j] = acc;
    }
}

} // namespace anp_hls

// ----- Top-level kernels (identical input interface) -----

// A. Original noisy-forward ANP: clean forward + full nonlinear noisy forward.
void anp_original_forward(
    const float x[16],
    const float W0[16 * 8], const float b0[8], const float eps0[8],
    const float W1[8 * 4],  const float b1[4], const float eps1[4],
    const float W2[4 * 2],  const float b2[2], const float eps2[2],
    float dz0[8], float dz1[4], float dz2[2],
    float clean_logits[2], float noisy_logits[2]);

// B. Dense linearized ANP: clean forward + first-order delta recurrence.
void anp_linearized_delta(
    const float x[16],
    const float W0[16 * 8], const float b0[8], const float eps0[8],
    const float W1[8 * 4],  const float b1[4], const float eps1[4],
    const float W2[4 * 2],  const float b2[2], const float eps2[2],
    float dz0[8], float dz1[4], float dz2[2],
    float clean_logits[2], float approx_logits[2]);

// C. Exact-original ANP with reused first layer: identical semantics/outputs to A,
//    but noisy layer-0 uses zn0 = zc0 + eps0 instead of recomputing x @ W0 + b0.
//    Later noisy layers stay full nonlinear. Same output interface as kernel A.
void anp_original_reuse_l0(
    const float x[16],
    const float W0[16 * 8], const float b0[8], const float eps0[8],
    const float W1[8 * 4],  const float b1[4], const float eps1[4],
    const float W2[4 * 2],  const float b2[2], const float eps2[2],
    float dz0[8], float dz1[4], float dz2[2],
    float clean_logits[2], float noisy_logits[2]);

#endif // ANP_HLS_HPP
