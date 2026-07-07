// Phase-1 portable CPU reference: compile-time network configuration.
//
// Debug network: 16 -> 8 -> 4 -> 2, batch size 1.
// Layout: row-major, weights [in, units], forward z = x @ W + b.
// Hidden activation: leaky ReLU, alpha = 0.2. Output: softmax.
//
// No HLS pragmas here. This is a plain-C++ scalar reference used to validate
// against the Phase-0 golden binaries before any Vitis HLS work.
#ifndef ANP_CONFIG_HPP
#define ANP_CONFIG_HPP

namespace anp {

// ----- Network dimensions (compile-time) -----
constexpr int N_LAYERS   = 3;
constexpr int BATCH      = 1;

constexpr int INPUT_DIM  = 16;
constexpr int OUTPUT_DIM = 2;

// Per-layer (in_dim, units).
constexpr int IN0 = 16, U0 = 8;   // layer 0: 16 -> 8   (leaky_relu)
constexpr int IN1 = 8,  U1 = 4;   // layer 1: 8  -> 4   (leaky_relu)
constexpr int IN2 = 4,  U2 = 2;   // layer 2: 4  -> 2   (softmax)

// ----- Activation -----
constexpr float LEAKY_ALPHA = 0.2f;

// ----- Validation -----
// Absolute tolerance for comparing the C++ reference to the TF golden vectors.
// Kept at 1e-6f per the Phase-1 brief; do NOT relax silently.
constexpr float DEFAULT_TOL = 1e-6f;

} // namespace anp

#endif // ANP_CONFIG_HPP
