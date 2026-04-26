// =============================================================================
// sliding_window.v
// =============================================================================
// Converts the vertical column output of the line_buffer into a full
// NUM_ROWS × WIN_SIZE 2-D patch by shifting each row through a horizontal
// shift-register of depth WIN_SIZE.
//
// Result: a flat bus containing NUM_ROWS*WIN_SIZE pixels — the complete
// neighbourhood patch centred (trailing-edge) around the current pixel.
//
// Parameters:
//   DATA_WIDTH – bits per pixel  (default 8)
//   WIN_SIZE   – window width/height (default 5)
//
// Interface:
//   clk        – system clock
//   rst        – synchronous active-high reset
//   col_in     – column bus from line_buffer: {row4..row0} each DATA_WIDTH wide
//   wr_en      – must be held high whenever col_in is valid (same signal that
//                drives line_buffer.wr_en)
//   window_out – flat 2-D patch: pixel[r][c] is at bit
//                  (r*WIN_SIZE + c + 1)*DATA_WIDTH - 1 : (r*WIN_SIZE + c)*DATA_WIDTH
//                  r=0 is the oldest row, c=0 is the oldest column.
//   valid_out  – high once the first WIN_SIZE pixels have been shifted in
//                (AND the line_buffer's valid_out is also high upstream)
// =============================================================================

`timescale 1ns / 1ps

module sliding_window #(
    parameter DATA_WIDTH = 8,
    parameter WIN_SIZE   = 5,
    parameter NUM_ROWS   = 5
)(
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire [NUM_ROWS*DATA_WIDTH-1:0]            col_in,
    input  wire                                      wr_en,
    output wire [NUM_ROWS*WIN_SIZE*DATA_WIDTH-1:0]   window_out,
    output reg                                        valid_out
);

    // -------------------------------------------------------------------------
    // Horizontal shift registers — one per row
    // Each is WIN_SIZE pixels deep.
    //
    // The shift-and-insert logic is inside a generate block so that the row
    // index (gr) is a genvar — a constant at elaboration time.  Vivado
    // requires bit-select bounds to be constants, so an integer loop variable
    // cannot be used as a slice index directly in an always block.
    // -------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] shift_reg [0:NUM_ROWS-1][0:WIN_SIZE-1];

    genvar gr_sh;
    generate
        for (gr_sh = 0; gr_sh < NUM_ROWS; gr_sh = gr_sh + 1) begin : gen_shift_row

            // Constant-width slice for this row extracted as a wire so the
            // bounds are unambiguously constant to the synthesiser.
            wire [DATA_WIDTH-1:0] col_pixel;
            assign col_pixel = col_in[(gr_sh+1)*DATA_WIDTH-1 : gr_sh*DATA_WIDTH];

            integer c_i;
            always @(posedge clk) begin
                if (rst) begin
                    for (c_i = 0; c_i < WIN_SIZE; c_i = c_i + 1)
                        shift_reg[gr_sh][c_i] <= 0;
                end else if (wr_en) begin
                    // Shift left: index 0 = oldest, WIN_SIZE-1 = newest
                    for (c_i = 0; c_i < WIN_SIZE - 1; c_i = c_i + 1)
                        shift_reg[gr_sh][c_i] <= shift_reg[gr_sh][c_i + 1];
                    // Insert newest pixel (constant-slice via the wire above)
                    shift_reg[gr_sh][WIN_SIZE-1] <= col_pixel;
                end
            end

        end
    endgenerate

    // -------------------------------------------------------------------------
    // Valid counter: need WIN_SIZE clocks of input before the window is full
    // -------------------------------------------------------------------------
    localparam CNT_W = $clog2(WIN_SIZE + 1);
    reg [CNT_W-1:0] px_cnt;

    always @(posedge clk) begin
        if (rst) begin
            px_cnt    <= 0;
            valid_out <= 1'b0;
        end else if (wr_en) begin
            if (px_cnt < WIN_SIZE) begin
                px_cnt <= px_cnt + 1;
            end
            if (px_cnt == WIN_SIZE - 1) begin
                valid_out <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Flatten shift registers onto output bus
    // window_out[row r, col c] = shift_reg[r][c]
    // -------------------------------------------------------------------------
    genvar gr, gc;
    generate
        for (gr = 0; gr < NUM_ROWS; gr = gr + 1) begin : gen_row
            for (gc = 0; gc < WIN_SIZE; gc = gc + 1) begin : gen_col
                assign window_out[((gr*WIN_SIZE + gc)+1)*DATA_WIDTH - 1 :
                                   (gr*WIN_SIZE + gc)  *DATA_WIDTH] =
                            shift_reg[gr][gc];
            end
        end
    endgenerate

endmodule
