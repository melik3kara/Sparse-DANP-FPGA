# Phase-2 HLS Step 3: 100 MHz synthesis + RTL co-simulation for kernel B.
# Dense linearized ANP, 100 MHz (10 ns), K26/KR260 part (xck26-sfvc784-2LV-c).
# Conservative baseline directives (no unroll/partition/pipeline/dataflow),
# identical treatment to kernel A.
#
# Sources are compiled in place from the space-free repository at ANP_HLS_ROOT.
# cosim uses the single-kernel tb_linearized.cpp (calls only anp_linearized_delta),
# which the shared tb_anp_hls.cpp cannot do (it calls both kernels).
set ROOT $env(ANP_HLS_ROOT)
set SRC $ROOT/src
set TB  $ROOT/tb

open_project -reset $ROOT/build/anp_linearized_prj
set_top anp_linearized_delta
add_files $SRC/anp_linearized.cpp
add_files -tb $TB/tb_linearized.cpp -cflags "-I$SRC"

open_solution -reset sol_100mhz
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

csynth_design
cosim_design
exit
