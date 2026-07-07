// Phase-1 testbench: validate the portable C++ reference against the Phase-0
// golden binaries (little-endian float32) for the 16 -> 8 -> 4 -> 2 network.
//
// Host-side file loading uses standard C++ utilities (this is allowed by the
// Phase-1 brief). The mathematical reference itself (anp_reference.cpp) uses no
// STL containers and no dynamic allocation.
//
// Exit code: 0 if every golden comparison passes at the tolerance; non-zero if
// any load fails or any comparison exceeds the tolerance.
#include "anp_reference.hpp"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>

using anp::real;

namespace {

std::string g_bin_dir;  // directory containing the golden *.bin files

// Read exactly `count` little-endian float32 values from <dir>/<name>.bin.
// Verifies the file size before reading. Returns false on any error.
bool load_bin(const char* name, real* buf, int count) {
    std::string path = g_bin_dir + "/" + name + ".bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("  [LOAD-FAIL] cannot open %s\n", path.c_str());
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    long expected = static_cast<long>(count) * static_cast<long>(sizeof(float));
    if (bytes != expected) {
        std::printf("  [LOAD-FAIL] %s: size %ld bytes, expected %ld (%d float32)\n",
                    path.c_str(), bytes, expected, count);
        std::fclose(f);
        return false;
    }
    // Host is little-endian (x86); float32 layout matches the file byte order.
    size_t got = std::fread(buf, sizeof(float), static_cast<size_t>(count), f);
    std::fclose(f);
    if (got != static_cast<size_t>(count)) {
        std::printf("  [LOAD-FAIL] %s: read %zu of %d floats\n", path.c_str(), got, count);
        return false;
    }
    return true;
}

int g_fail_count = 0;
int g_pass_count = 0;

// Compare a computed tensor against its golden binary; print name/count/max-err/verdict.
void compare(const char* golden_name, const real* computed, int count, real tol) {
    real golden[64];   // max tensor here is 16 elements; 64 is a safe fixed cap
    if (count > 64) { std::printf("  [ERR] tensor %s too large for buffer\n", golden_name); g_fail_count++; return; }
    if (!load_bin(golden_name, golden, count)) { g_fail_count++; return; }

    real max_err = 0.0f;
    for (int i = 0; i < count; ++i) {
        real e = std::fabs(computed[i] - golden[i]);
        if (e > max_err) max_err = e;
    }
    bool pass = (max_err <= tol);
    std::printf("  %-16s  n=%-3d  max_abs_err=%.3e  %s\n",
                golden_name, count, max_err, pass ? "PASS" : "FAIL");
    if (pass) ++g_pass_count; else ++g_fail_count;
}

// Max abs diff between two computed tensors (for the A-vs-B diagnostic).
real max_abs_diff(const real* a, const real* b, int count) {
    real m = 0.0f;
    for (int i = 0; i < count; ++i) {
        real e = std::fabs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

} // namespace

int main(int argc, char** argv) {
    g_bin_dir = (argc > 1) ? argv[1] : "../phase0/golden_debug_16_8_4_2/bin";
    const real tol = anp::DEFAULT_TOL;

    std::printf("Phase-1 C++ reference vs Phase-0 golden vectors\n");
    std::printf("golden dir : %s\n", g_bin_dir.c_str());
    std::printf("tolerance  : %.1e (absolute)\n\n", tol);

    // ---- Load kernel inputs from the golden binaries ----
    anp::Inputs in;
    bool ok = true;
    ok &= load_bin("input", in.input, anp::INPUT_DIM);
    ok &= load_bin("W0", in.W0, anp::IN0 * anp::U0); ok &= load_bin("b0", in.b0, anp::U0); ok &= load_bin("eps0", in.eps0, anp::U0);
    ok &= load_bin("W1", in.W1, anp::IN1 * anp::U1); ok &= load_bin("b1", in.b1, anp::U1); ok &= load_bin("eps1", in.eps1, anp::U1);
    ok &= load_bin("W2", in.W2, anp::IN2 * anp::U2); ok &= load_bin("b2", in.b2, anp::U2); ok &= load_bin("eps2", in.eps2, anp::U2);
    if (!ok) {
        std::printf("\n[FATAL] could not load kernel inputs. Is the golden dir correct?\n");
        return 2;
    }

    // ---- Compute both reference paths ----
    anp::Results r;
    anp::compute_reference(in, r);

    // ---- A. Original noisy-forward ANP ----
    std::printf("=== A. Original noisy-forward ANP ===\n");
    compare("z_clean_0",   r.z_clean0, anp::U0, tol);
    compare("z_clean_1",   r.z_clean1, anp::U1, tol);
    compare("z_clean_2",   r.z_clean2, anp::U2, tol);
    compare("a_clean_0",   r.a_clean0, anp::U0, tol);
    compare("a_clean_1",   r.a_clean1, anp::U1, tol);
    compare("a_clean_2",   r.a_clean2, anp::U2, tol);
    compare("z_noisy_0",   r.z_noisy0, anp::U0, tol);
    compare("z_noisy_1",   r.z_noisy1, anp::U1, tol);
    compare("z_noisy_2",   r.z_noisy2, anp::U2, tol);
    compare("a_noisy_0",   r.a_noisy0, anp::U0, tol);
    compare("a_noisy_1",   r.a_noisy1, anp::U1, tol);
    compare("a_noisy_2",   r.a_noisy2, anp::U2, tol);
    compare("dz_original_0", r.dz_orig0, anp::U0, tol);
    compare("dz_original_1", r.dz_orig1, anp::U1, tol);
    compare("dz_original_2", r.dz_orig2, anp::U2, tol);
    compare("clean_logits", r.clean_logits, anp::OUTPUT_DIM, tol);
    compare("noisy_logits", r.noisy_logits, anp::OUTPUT_DIM, tol);
    compare("clean_probs",  r.clean_probs,  anp::OUTPUT_DIM, tol);
    compare("noisy_probs",  r.noisy_probs,  anp::OUTPUT_DIM, tol);

    // ---- B. Dense linearized ANP ----
    std::printf("\n=== B. Dense linearized ANP ===\n");
    compare("dz_linearized_0", r.dz_lin0, anp::U0, tol);
    compare("dz_linearized_1", r.dz_lin1, anp::U1, tol);
    compare("dz_linearized_2", r.dz_lin2, anp::U2, tol);
    compare("da_linearized_0", r.da_lin0, anp::U0, tol);
    compare("da_linearized_1", r.da_lin1, anp::U1, tol);
    compare("approx_logits",   r.approx_logits, anp::OUTPUT_DIM, tol);
    compare("approx_probs",    r.approx_probs,  anp::OUTPUT_DIM, tol);

    // ---- A-vs-B diagnostic (C++ side; not a golden comparison) ----
    real dz_AvsB = 0.0f;
    dz_AvsB = std::fmax(dz_AvsB, max_abs_diff(r.dz_orig0, r.dz_lin0, anp::U0));
    dz_AvsB = std::fmax(dz_AvsB, max_abs_diff(r.dz_orig1, r.dz_lin1, anp::U1));
    dz_AvsB = std::fmax(dz_AvsB, max_abs_diff(r.dz_orig2, r.dz_lin2, anp::U2));
    real logits_AvsB = max_abs_diff(r.noisy_logits, r.approx_logits, anp::OUTPUT_DIM);
    std::printf("\n=== A-vs-B diagnostic (C++) ===\n");
    std::printf("  max_abs_diff dz_original vs dz_linearized : %.3e\n", dz_AvsB);
    std::printf("  max_abs_diff noisy_logits vs approx_logits: %.3e\n", logits_AvsB);
    // Interpretation depends on whether the perturbation crossed a Leaky-ReLU
    // boundary in this fixture. 1e-4 separates float32-level noise from a real
    // approximation gap (nominal ~7e-8; branch-crossing ~4e-3).
    if (dz_AvsB < 1e-4f) {
        std::printf("  NOTE: near float32-level -> this fixture's perturbations do NOT cross\n");
        std::printf("        Leaky-ReLU branch boundaries. Validates implementation\n");
        std::printf("        correctness, NOT general ANP-vs-linearized equivalence.\n");
    } else {
        std::printf("  NOTE: well above float32-level -> at least one Leaky-ReLU branch\n");
        std::printf("        crossing occurred; this is a GENUINE original-vs-linearized\n");
        std::printf("        approximation gap (frozen f'(z_clean) misses the branch flip).\n");
    }

    // ---- Summary / exit code ----
    std::printf("\n=== Summary ===\n");
    std::printf("  passed: %d   failed: %d\n", g_pass_count, g_fail_count);
    if (g_fail_count != 0) {
        std::printf("[RESULT] FAIL (%d tensor(s) exceeded tol=%.1e)\n", g_fail_count, tol);
        return 1;
    }
    std::printf("[RESULT] PASS (all %d golden comparisons within tol=%.1e)\n", g_pass_count, tol);
    return 0;
}
