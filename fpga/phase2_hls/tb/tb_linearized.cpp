// Phase-2 HLS per-kernel testbench: anp_linearized_delta ONLY.
//
// Single-kernel testbench required for RTL co-simulation: cosim wraps only the
// synthesis top in RTL, so the testbench must call exactly one kernel (the shared
// tb_anp_hls.cpp calls both and cannot be used for cosim).
//
// Drives anp_linearized_delta against BOTH Phase-0 golden fixtures
// (nominal + branch_crossing), comparing every kernel output to its golden binary
// at absolute tolerance 1e-6. Returns non-zero on any mismatch or load failure.
//
// Golden-fixture root resolved at RUNTIME from env var ANP_GOLDEN_ROOT (the same
// value used by C-simulation); absolute path so cosim's working directory is
// irrelevant. Host-only file I/O (allowed in the TB).
#include "anp_hls.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#ifndef GOLDEN_ROOT_DEFAULT
#define GOLDEN_ROOT_DEFAULT "../phase0"
#endif

namespace {

const float TOL = 1e-6f;
int g_pass = 0, g_fail = 0;

std::string golden_root() {
    const char* e = std::getenv("ANP_GOLDEN_ROOT");
    return (e && e[0]) ? std::string(e) : std::string(GOLDEN_ROOT_DEFAULT);
}

bool load_bin(const std::string& dir, const char* name, float* buf, int count) {
    std::string path = dir + "/" + name + ".bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("    [LOAD-FAIL] cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    long expected = static_cast<long>(count) * static_cast<long>(sizeof(float));
    if (bytes != expected) {
        std::printf("    [LOAD-FAIL] %s: %ld bytes, expected %ld\n", path.c_str(), bytes, expected);
        std::fclose(f); return false;
    }
    size_t got = std::fread(buf, sizeof(float), static_cast<size_t>(count), f);
    std::fclose(f);
    if (got != static_cast<size_t>(count)) {
        std::printf("    [LOAD-FAIL] %s: short read\n", path.c_str()); return false;
    }
    return true;
}

void compare(const std::string& dir, const char* golden_name,
             const float* computed, int count) {
    float golden[64];
    if (count > 64) { std::printf("    [ERR] %s too large\n", golden_name); g_fail++; return; }
    if (!load_bin(dir, golden_name, golden, count)) { g_fail++; return; }
    float max_err = 0.0f;
    for (int i = 0; i < count; ++i) {
        float e = std::fabs(computed[i] - golden[i]);
        if (e > max_err) max_err = e;
    }
    bool pass = (max_err <= TOL);
    std::printf("    %-16s n=%-3d max_abs_err=%.3e  %s\n",
                golden_name, count, max_err, pass ? "PASS" : "FAIL");
    if (pass) ++g_pass; else ++g_fail;
}

struct Inputs {
    float x[16];
    float W0[16 * 8], b0[8], eps0[8];
    float W1[8 * 4],  b1[4], eps1[4];
    float W2[4 * 2],  b2[2], eps2[2];
};

bool load_inputs(const std::string& dir, Inputs& in) {
    bool ok = true;
    ok &= load_bin(dir, "input", in.x, 16);
    ok &= load_bin(dir, "W0", in.W0, 16 * 8); ok &= load_bin(dir, "b0", in.b0, 8); ok &= load_bin(dir, "eps0", in.eps0, 8);
    ok &= load_bin(dir, "W1", in.W1, 8 * 4);  ok &= load_bin(dir, "b1", in.b1, 4); ok &= load_bin(dir, "eps1", in.eps1, 4);
    ok &= load_bin(dir, "W2", in.W2, 4 * 2);  ok &= load_bin(dir, "b2", in.b2, 2); ok &= load_bin(dir, "eps2", in.eps2, 2);
    return ok;
}

// Exercise the linearized kernel on one fixture.
void run_fixture(const std::string& fixture_subdir) {
    std::string dir = golden_root() + "/" + fixture_subdir + "/bin";
    std::printf("\n================ fixture: %s ================\n", fixture_subdir.c_str());

    Inputs in;
    if (!load_inputs(dir, in)) {
        std::printf("  [FATAL] could not load inputs from %s\n", dir.c_str());
        g_fail++;
        return;
    }

    float dz0[8], dz1[4], dz2[2], clean[2], approx[2];
    anp_linearized_delta(in.x, in.W0, in.b0, in.eps0, in.W1, in.b1, in.eps1,
                         in.W2, in.b2, in.eps2,
                         dz0, dz1, dz2, clean, approx);
    std::printf("  --- anp_linearized_delta ---\n");
    compare(dir, "dz_linearized_0", dz0, 8);
    compare(dir, "dz_linearized_1", dz1, 4);
    compare(dir, "dz_linearized_2", dz2, 2);
    compare(dir, "clean_logits",    clean, 2);
    compare(dir, "approx_logits",   approx, 2);
}

} // namespace

int main() {
    std::printf("Phase-2 HLS per-kernel TB: anp_linearized_delta vs Phase-0 golden\n");
    std::printf("ANP_GOLDEN_ROOT = %s   tolerance = %.1e\n", golden_root().c_str(), TOL);

    run_fixture("golden_debug_16_8_4_2");                    // nominal
    run_fixture("golden_stress_branch_crossing_16_8_4_2");   // branch-crossing

    std::printf("\n=== Summary ===  passed: %d  failed: %d\n", g_pass, g_fail);
    if (g_fail != 0) {
        std::printf("[RESULT] FAIL\n");
        return 1;
    }
    std::printf("[RESULT] PASS (all comparisons within tol=%.1e)\n", TOL);
    return 0;
}
