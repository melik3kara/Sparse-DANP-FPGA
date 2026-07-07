# Phase-2 HLS: C simulation only for kernel C (anp_original_reuse_l0).
# Validates against the ORIGINAL golden via the single-kernel tb_original_reuse.cpp.
# Design source added as -tb too so its definition compiles into csim.exe (csim-only flow).
set ROOT $env(ANP_HLS_ROOT)
set SRC $ROOT/src
set TB  $ROOT/tb

open_project -reset $ROOT/build/anp_original_reuse_csim_prj
open_solution -reset sol_csim
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

set_top anp_original_reuse_l0
add_files $SRC/anp_original_reuse.cpp
add_files -tb $TB/tb_original_reuse.cpp    -cflags "-I$SRC"
add_files -tb $SRC/anp_original_reuse.cpp  -cflags "-I$SRC"

csim_design
exit
