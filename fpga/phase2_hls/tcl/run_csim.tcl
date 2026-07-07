# Phase-2 HLS Step 2: C simulation only (no synthesis).
# Drives both kernels against both golden fixtures via tb/tb_anp_hls.cpp.
#
# Sources are compiled in place from the space-free repository at ANP_HLS_ROOT;
# the golden fixtures are exposed via ANP_GOLDEN_ROOT (read by the testbench at
# runtime with fopen).
#
# NOTE: under the batch open_project flow, csim links ONLY files added with -tb.
# For this csim-only step the design sources are therefore added as -tb as well
# so their definitions are compiled into csim.exe. The synthesis scripts
# (run_original.tcl / run_linearized.tcl) add them as proper design files.
set ROOT $env(ANP_HLS_ROOT)
set SRC $ROOT/src
set TB  $ROOT/tb

open_project -reset $ROOT/build/anp_csim_prj
open_solution -reset sol_csim
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

set_top anp_original_forward
add_files $SRC/anp_original.cpp
add_files $SRC/anp_linearized.cpp
add_files -tb $TB/tb_anp_hls.cpp     -cflags "-I$SRC"
add_files -tb $SRC/anp_original.cpp  -cflags "-I$SRC"
add_files -tb $SRC/anp_linearized.cpp -cflags "-I$SRC"

csim_design
exit
