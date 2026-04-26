# System clock: 100 MHz (10 ns period)
# Targets the clk port of stereo_top on the Artix-7 xc7a100tcsg324-1.
# Without this constraint Vivado cannot perform timing analysis and every
# flip-flop in the design produces a TIMING-17 "Non-clocked sequential cell"
# critical warning during implementation.
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

# I/O false paths
# This is a simulation-only design with no physical board I/O.
# Suppress TIMING-8 "missing input/output delay" methodology warnings by
# declaring false paths on all non-clock ports.  This tells Vivado not to
# check setup/hold timing at the chip boundary.

