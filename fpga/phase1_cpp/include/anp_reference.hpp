// Phase-1 portable CPU reference: public API.
//
// Two reference paths, computed from the same Inputs into one Results struct:
//   A. original noisy-forward ANP  (full nonlinear noisy forward)
//   B. dense linearized ANP        (first-order delta recurrence)
//
// The math functions use only fixed-size arrays and float arithmetic; no
// dynamic allocation, no STL containers, no external libraries.
#ifndef ANP_REFERENCE_HPP
#define ANP_REFERENCE_HPP

#include "anp_types.hpp"

namespace anp {

// ----- Elementwise scalar math (exposed for testing/reuse) -----
real leaky_relu(real z);
real leaky_relu_deriv(real z);   // 1 for z > 0, alpha otherwise (matches TF grad; z==0 avoided by fixture)

// Numerically stable softmax over `n` logits (subtracts the max logit).
void softmax(const real* logits, real* probs, int n);

// z[j] = b[j] + sum_i x[i] * W[i*units + j]   (row-major W, layout [in, units])
void dense(const real* x, const real* W, const real* b,
           real* z, int in_dim, int units);

// z[j] = eps[j] + sum_i da[i] * W[i*units + j]   (linearized propagation; bias cancels)
void matvec_add_eps(const real* da, const real* W, const real* eps,
                    real* z, int in_dim, int units);

// ----- Full reference paths -----
void compute_clean(const Inputs& in, Results& r);
void compute_original(const Inputs& in, Results& r);   // requires compute_clean first
void compute_linearized(const Inputs& in, Results& r); // requires compute_clean first

// Convenience: clean -> original -> linearized.
void compute_reference(const Inputs& in, Results& r);

} // namespace anp

#endif // ANP_REFERENCE_HPP
