# vivado_sim.tcl — create Vivado project, add hls4ml RTL, run behavioral simulation
#
# Usage (from workspace root):
#   source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh
#   vivado -mode batch -source v2/hls4ml/vivado_sim.tcl \
#          -log v2/hls4ml/vivado_sim.log \
#          -journal v2/hls4ml/vivado_sim.jou

set script_dir [file dirname [file normalize [info script]]]
set hls_dir    [file normalize "$script_dir/fixed16_6"]
set rtl_dir    "$hls_dir/myproject_prj/solution1/syn/verilog"
set tb_dir     "$hls_dir/tb_data"
set proj_name  "myproject_sim"
set proj_dir   [file normalize "$script_dir/vivado_sim_project"]
# Match the part used during HLS synthesis (project.tcl: xc7a200tfbg484-1, Artix-7 200T)
set part       "xc7a200tfbg484-1"

# ---- Validate prerequisites -------------------------------------------------
if {![file isdirectory $rtl_dir]} {
    puts "ERROR: RTL directory not found: $rtl_dir"
    puts "ERROR: Run C/RTL synthesis first (build_prj.tcl -synth)"
    exit 1
}

set tb_sv [file normalize "$script_dir/myproject_tb.sv"]
if {![file exists $tb_sv]} {
    puts "ERROR: Testbench not found: $tb_sv"
    exit 1
}

set hex_mem "$tb_dir/tb_input_hex.mem"
if {![file exists $hex_mem]} {
    puts "WARNING: $hex_mem not found — simulation will use zero inputs unless you"
    puts "WARNING: run gen_tb_vectors.py first."
}

# ---- Create project ---------------------------------------------------------
puts "INFO: Creating Vivado project '$proj_name' in $proj_dir"
create_project $proj_name $proj_dir -part $part -force
set_property simulator_language Mixed [current_project]
set_property target_simulator XSim [current_project]

# ---- Add all HLS-generated Verilog RTL sources ------------------------------
set rtl_files [glob -nocomplain "$rtl_dir/*.v"]
if {[llength $rtl_files] == 0} {
    puts "ERROR: No .v files found in $rtl_dir"
    exit 1
}
puts "INFO: Adding [llength $rtl_files] Verilog RTL file(s)"
add_files -norecurse $rtl_files
set_property file_type {Verilog} [get_files -filter {FILE_TYPE == "Verilog"}]

# Xilinx simulation libraries (glbl, etc.) — add if present
set glbl [file join $::env(XILINX_VIVADO) "data/verilog/src/glbl.v"]
if {[file exists $glbl]} {
    add_files -norecurse $glbl
}

# ---- Add simulation testbench -----------------------------------------------
add_files -fileset sim_1 -norecurse $tb_sv

# Set simulation top
set_property top myproject_tb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# ---- XSim simulation settings -----------------------------------------------
# Disable runtime limit in the property — we drive time via 'run' command below
set_property -name {xsim.simulate.runtime} -value {0} \
    -objects [get_filesets sim_1]
set_property -name {xsim.simulate.log_all_signals} -value {false} \
    -objects [get_filesets sim_1]

# ---- Stage tb_data/ into the xsim working directory -------------------------
# xsim runs from: <proj_dir>/<proj_name>.sim/sim_1/behav/xsim/
# $readmemh / $fopen paths in the testbench are relative to that directory.
set xsim_rundir [file normalize "$proj_dir/${proj_name}.sim/sim_1/behav/xsim"]
file mkdir $xsim_rundir
if {[file isdirectory $tb_dir]} {
    # Copy tb_data/ so $readmemh("tb_data/tb_input_hex.mem") resolves correctly
    if {[file exists "$xsim_rundir/tb_data"]} {
        file delete -force "$xsim_rundir/tb_data"
    }
    file copy $tb_dir $xsim_rundir/tb_data
    puts "INFO: Staged tb_data/ → $xsim_rundir/tb_data"
} else {
    puts "WARNING: tb_data not found at $tb_dir — \$readmemh will fail"
}

# ---- Launch behavioral simulation -------------------------------------------
puts "INFO: Launching behavioral simulation (xsim) — may take 20 min per frame"
launch_simulation -simset sim_1 -mode behavioral

# Run until $finish (testbench calls $finish after all frames complete).
# 400 ms covers up to 20 frames × ~18 min each (wall time, not sim time).
run 400 ms

# ---- Save waveform configuration -------------------------------------------
set wcfg_path "$proj_dir/${proj_name}.wcfg"
if {[catch {save_wave_config $wcfg_path} err]} {
    puts "WARNING: Could not save wave config: $err"
} else {
    puts "INFO: Waveform config saved → $wcfg_path"
}

puts "INFO: Simulation complete"
close_sim
close_project
exit
