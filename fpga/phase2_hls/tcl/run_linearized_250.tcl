# Phase-2 HLS Step 3: 250 MHz synthesis-ONLY characterization for kernel B.
# Dense linearized ANP, 250 MHz (4 ns), K26/KR260 part (xck26-sfvc784-2LV-c).
# Identical code/interfaces/directives to the 100 MHz run; only the clock differs.
# Separate project so the 100 MHz results are never overwritten. No cosim here.
set ROOT $env(ANP_HLS_ROOT)
set SRC $ROOT/src
set TB  $ROOT/tb

open_project -reset $ROOT/build/anp_linearized_250_prj
set_top anp_linearized_delta
add_files $SRC/anp_linearized.cpp
add_files -tb $TB/tb_linearized.cpp -cflags "-I$SRC"

open_solution -reset sol_250mhz
set_part {xck26-sfvc784-2LV-c}
create_clock -period 4 -name default

csynth_design
exit
