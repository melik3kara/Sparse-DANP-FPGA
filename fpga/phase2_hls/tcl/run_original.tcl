# Phase-2 HLS Step 3: 100 MHz synthesis + RTL co-simulation for kernel A.
# Original noisy-forward ANP, 100 MHz (10 ns), K26/KR260 part (xck26-sfvc784-2LV-c).
# Conservative baseline directives (no unroll/partition/pipeline/dataflow).
#
# Sources are compiled in place from the space-free repository at ANP_HLS_ROOT.
# cosim uses the single-kernel tb_original.cpp (calls only anp_original_forward),
# which the shared tb_anp_hls.cpp cannot do (it calls both kernels).
set ROOT $env(ANP_HLS_ROOT)
set SRC $ROOT/src
set TB  $ROOT/tb

open_project -reset $ROOT/build/anp_original_prj
set_top anp_original_forward
add_files $SRC/anp_original.cpp
add_files -tb $TB/tb_original.cpp -cflags "-I$SRC"

open_solution -reset sol_100mhz
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

csynth_design
cosim_design
exit
