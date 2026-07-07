# Phase-2 HLS: 100 MHz synthesis + RTL co-simulation for kernel C.
# Exact-original ANP with reused first layer, 100 MHz (10 ns), part xck26-sfvc784-2LV-c.
# Conservative baseline directives (no unroll/partition/pipeline/dataflow), identical
# treatment to kernels A and B. Separate project so kernel A/B results are untouched.
#
# cosim uses the single-kernel tb_original_reuse.cpp (calls only anp_original_reuse_l0),
# validated against the ORIGINAL golden.
set ROOT $env(ANP_HLS_ROOT)
set SRC $ROOT/src
set TB  $ROOT/tb

open_project -reset $ROOT/build/anp_original_reuse_prj
set_top anp_original_reuse_l0
add_files $SRC/anp_original_reuse.cpp
add_files -tb $TB/tb_original_reuse.cpp -cflags "-I$SRC"

open_solution -reset sol_100mhz
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

csynth_design
cosim_design
exit
