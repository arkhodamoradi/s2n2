open_project hls-syn-dup_stream
add_files dup_stream_top.cpp -cflags "-std=c++0x -I$::env(FINN_HLS_ROOT) -I$::env(FINN_HLS_ROOT)/tb" 
add_files -tb dup_stream_tb.cpp -cflags "-std=c++0x -I$::env(FINN_HLS_ROOT) -I$::env(FINN_HLS_ROOT)/tb" 
set_top Testbench_dup_stream
open_solution sol1
set_part {xczu3eg-sbva484-1-i}
create_clock -period 5 -name default
csim_design
csynth_design
cosim_design
exit
