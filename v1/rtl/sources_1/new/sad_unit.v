// =============================================================================
// sad_unit.v  (resource-optimised)
// =============================================================================
// Computes SAD between two 5×5 patches using a SEQUENTIAL accumulator.
//
// Instead of 25 parallel subtractors + a 25-operand adder tree (expensive),
// this version uses ONE subtractor and ONE adder that iterate over all 25
// pixel pairs across 25 clock cycles.  LUT cost drops from ~750 to ~30.
//
// Latency: 25 + 1 = 26 cycles from valid_in to valid_out.
// Throughput: one SAD result every 26 cycles.  stereo_top drives this with
// one call per output pixel (one SAD per pixel position, cycling through all
// MAX_DISP disparities in turn), so throughput is fully adequate.
//
// Parameters:
//   DATA_WIDTH  – bits per pixel  (default 8)
//   WIN_SIZE    – window width/height (default 5)
//
// Interface:
//   clk         – system clock
//   rst         – synchronous active-high reset
//   patch_left  – flat 5×5 patch from left image  (25 × DATA_WIDTH bits)
//   patch_right – flat 5×5 patch from right image (25 × DATA_WIDTH bits)
//   valid_in    – strobe: patches are valid THIS cycle, begin accumulation
//   sad_out     – accumulated SAD (13-bit: max 255×25=6375)
//   valid_out   – sad_out is valid (pulses for one cycle)
// =============================================================================

`timescale 1ns / 1ps

module sad_unit #(
    parameter DATA_WIDTH = 8,
    parameter WIN_SIZE   = 5
)(
    input  wire                                    clk,
    input  wire                                    rst,
    input  wire [WIN_SIZE*WIN_SIZE*DATA_WIDTH-1:0] patch_left,
    input  wire [WIN_SIZE*WIN_SIZE*DATA_WIDTH-1:0] patch_right,
    input  wire                                    valid_in,
    output reg  [12:0]                             sad_out,
    output reg                                     valid_out
);

    localparam NUM_PIXELS = WIN_SIZE * WIN_SIZE;   // 25
    localparam PIX_CNT_W  = $clog2(NUM_PIXELS);   // 5 bits

    // -------------------------------------------------------------------------
    // Internal registers
    // -------------------------------------------------------------------------
    reg                  busy;          // accumulation in progress
    reg [PIX_CNT_W-1:0]  pix_idx;       // current pixel being processed (0..24)
    reg [12:0]           accum;         // running sum

    // Latch the two patches at the start of a new accumulation so they remain
    // stable throughout the 25-cycle window.
    reg [WIN_SIZE*WIN_SIZE*DATA_WIDTH-1:0] latch_left;
    reg [WIN_SIZE*WIN_SIZE*DATA_WIDTH-1:0] latch_right;

    // -------------------------------------------------------------------------
    // Extract one pixel pair using the current index.
    // pix_idx is a register, so the slice bounds are NOT constants — we must
    // use a variable-select.  Verilog's indexed part-select (+: operator)
    // accepts a variable base with a constant width, which is synthesis-safe.
    // -------------------------------------------------------------------------
    wire [DATA_WIDTH-1:0] lp = latch_left [pix_idx*DATA_WIDTH +: DATA_WIDTH];
    wire [DATA_WIDTH-1:0] rp = latch_right[pix_idx*DATA_WIDTH +: DATA_WIDTH];

    // Absolute difference — one 9-bit subtractor + sign-conditional negate
    wire [DATA_WIDTH:0] sub  = {1'b0, lp} - {1'b0, rp};
    wire [DATA_WIDTH:0] absd = sub[DATA_WIDTH] ? (~sub + 1'b1) : sub;

    // -------------------------------------------------------------------------
    // FSM / accumulator
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            busy      <= 1'b0;
            pix_idx   <= 0;
            accum     <= 0;
            sad_out   <= 0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= 1'b0;   // default: not valid

            if (!busy && valid_in) begin
                // --- Start of a new SAD computation ---
                latch_left  <= patch_left;
                latch_right <= patch_right;
                busy        <= 1'b1;
                pix_idx     <= 0;
                accum       <= 0;

            end else if (busy) begin
                // --- Accumulate one pixel per cycle ---
                if (pix_idx == NUM_PIXELS - 1) begin
                    // Last pixel: register result and signal done
                    sad_out   <= accum + absd;
                    valid_out <= 1'b1;
                    busy      <= 1'b0;
                end else begin
                    accum   <= accum + absd;
                    pix_idx <= pix_idx + 1;
                end
            end
        end
    end

endmodule
