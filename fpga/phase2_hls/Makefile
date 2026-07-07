# Phase-2 HLS driver.
#
# Requires the Vitis 2025.1 environment on PATH first:
#   source /home/melike/Xilinx/2025.1/Vitis/settings64.sh
#
# The repository now lives at a space-free path, so the HLS sources are compiled
# directly in place -- no external staging copy and no golden-fixture symlink.
# The testbench reads the golden root at runtime via ANP_GOLDEN_ROOT.
# Generated projects go under build/ and run logs under logs/ (both gitignored).

VITISRUN     ?= vitis-run --mode hls --tcl
GOLDEN_ROOT  := $(abspath ../phase0)
BUILD        := $(CURDIR)/build
LOGS         := $(CURDIR)/logs
RUNENV        = ANP_HLS_ROOT="$(CURDIR)" ANP_GOLDEN_ROOT="$(GOLDEN_ROOT)"

.PHONY: csim csim_reuse syn100 syn_original syn_linearized syn_original_reuse syn250 syn_original_250 syn_linearized_250 clean distclean

# Step 2: C simulation only, both kernels, both fixtures (shared testbench).
# Runs from build/ so stray vitis logs stay out of the repo; a copy of the run
# output is teed into logs/csim.log.
csim:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_csim.tcl" 2>&1 | tee "$(LOGS)/csim.log"

# Audit: C simulation only for the exact-reuse kernel C (validated vs original golden).
csim_reuse:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_csim_reuse.tcl" 2>&1 | tee "$(LOGS)/csim_reuse.log"

# Step 3: 100 MHz (10 ns) synthesis + RTL co-simulation, per kernel, separate
# projects. Each uses its single-kernel testbench so cosim can wrap one top.
syn100: syn_original syn_linearized

syn_original:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_original.tcl" 2>&1 | tee "$(LOGS)/syn_original_100mhz.log"

syn_linearized:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_linearized.tcl" 2>&1 | tee "$(LOGS)/syn_linearized_100mhz.log"

# Audit: 100 MHz synth + cosim for exact-reuse kernel C (separate project).
syn_original_reuse:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_original_reuse.tcl" 2>&1 | tee "$(LOGS)/syn_original_reuse_100mhz.log"

# Step 3: 250 MHz (4 ns) synthesis-only characterization, per kernel, in separate
# projects so the 100 MHz results are never overwritten.
syn250: syn_original_250 syn_linearized_250

syn_original_250:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_original_250.tcl" 2>&1 | tee "$(LOGS)/syn_original_250mhz.log"

syn_linearized_250:
	mkdir -p "$(BUILD)" "$(LOGS)"
	cd "$(BUILD)" && $(RUNENV) $(VITISRUN) "$(CURDIR)/tcl/run_linearized_250.tcl" 2>&1 | tee "$(LOGS)/syn_linearized_250mhz.log"

clean:
	rm -f vitis_hls.log vitis-run.log
	rm -rf "$(BUILD)"

distclean: clean
	rm -rf "$(LOGS)"
