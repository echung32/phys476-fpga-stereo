// =============================================================================
// stereo_top.v  (v3 — interleaved pixel stream)
// =============================================================================
// Top-level stereo depth module.
//
// INPUT STREAM ORDER (IMPORTANT):
//   INTERLEAVED: for each column c in each row, right[c] then left[c].
//   This ensures that when the FSM snapshots at left column c, the right-eye
//   delay chain holds patches for right cols c, c-1, ..., c-d (d=0..MAX_DISP-1).
//
//   The previous "half-row" layout (all right pixels then all left pixels)
//   caused the delay chain to be misaligned: when processing left col c, the
//   chain held patches from right cols 511, 510, ... instead of c, c-1, ...
//
// Architecture:
//  • Single sad_unit, time-multiplexed over MAX_DISP disparities.
//  • When the FSM is processing a pixel (SNAP/RUNNING), it deasserts `ready`,
//    stalling the input stream (valid/ready handshake).
//  • Pixel splitter: col_cnt[0]=0 → right (even slot), col_cnt[0]=1 → left (odd slot).
//
// Interface:
//   clk        – system clock
//   rst        – synchronous active-high reset
//   pixel_in   – 8-bit grayscale pixel
//   pix_valid  – input valid strobe
//   ready      – output: pipeline accepts a pixel this cycle
//   disp_out   – disparity result (MAX_DISP levels, $clog2(MAX_DISP) bits)
//   disp_valid – disparity output valid strobe
// =============================================================================

`timescale 1ns / 1ps

module stereo_top #(
    parameter DATA_WIDTH = 8,
    parameter IMG_WIDTH  = 512,
    parameter WIN_SIZE   = 5,
    parameter NUM_ROWS   = 5,
    parameter MAX_DISP   = 16
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire [DATA_WIDTH-1:0]       pixel_in,
    input  wire                        pix_valid,
    output wire                        ready,         // NEW: pipeline accepts input
    output reg  [$clog2(MAX_DISP)-1:0] disp_out,
    output reg                         disp_valid
);

    localparam DISP_W    = $clog2(MAX_DISP);          // 4
    localparam PATCH_W   = NUM_ROWS * WIN_SIZE * DATA_WIDTH;  // 200 bits

    // =========================================================================
    // FSM states
    //   ST_IDLE    – accepting pixels; waiting for a left-pixel trigger
    //   ST_HOLD    – 3-cycle pipeline bubble to let BRAM read settle
    //   ST_SNAP    – latch patch data and start disparity 0
    //   ST_RUNNING – iterate through MAX_DISP disparities
    // =========================================================================
    localparam ST_IDLE    = 2'd0;
    localparam ST_HOLD    = 2'd1;
    localparam ST_SNAP    = 2'd2;
    localparam ST_RUNNING = 2'd3;

    reg [1:0] state;
    reg [1:0] hold_cnt;   // 2-bit counter for the 3-cycle hold

    // Pipeline stalls whenever the FSM is not in IDLE
    assign ready = (state == ST_IDLE);

    // =========================================================================
    // 1. Pixel splitter
    //    INTERLEAVED layout: right pixel first (col_cnt[0]=0), left second (col_cnt[0]=1).
    //    col_cnt wraps at 2*IMG_WIDTH (one full interleaved pair-row = 2*512 slots).
    // =========================================================================
    localparam COL_W = $clog2(2 * IMG_WIDTH);   // 10

    reg [COL_W-1:0] col_cnt;
    wire pix_accepted = pix_valid & ready;

    always @(posedge clk) begin
        if (rst)
            col_cnt <= 0;
        else if (pix_accepted)
            col_cnt <= (col_cnt == 2*IMG_WIDTH - 1) ? 0 : col_cnt + 1;
    end

    wire is_right = ~col_cnt[0];  // even slot → right eye
    wire is_left  =  col_cnt[0];  // odd slot  → left eye

    wire left_wr_en  = pix_accepted & is_left;
    wire right_wr_en = pix_accepted & is_right;

    // =========================================================================
    // 2a. Left eye: line_buffer → sliding_window
    // =========================================================================
    wire [NUM_ROWS*DATA_WIDTH-1:0]  left_col;
    wire                             left_lb_valid;
    wire                             left_lb_wr_en_out;

    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_WIDTH (IMG_WIDTH),
        .NUM_ROWS  (NUM_ROWS)
    ) u_left_lb (
        .clk         (clk),
        .rst         (rst),
        .pixel_in    (pixel_in),
        .wr_en       (left_wr_en),
        .pixel_out   (left_col),
        .valid_out   (left_lb_valid),
        .wr_en_out   (left_lb_wr_en_out),
        .wr_en_d1_out()                     // unused for left eye
    );

    wire [PATCH_W-1:0] patch_left;
    wire               left_sw_valid;

    sliding_window #(
        .DATA_WIDTH(DATA_WIDTH),
        .WIN_SIZE  (WIN_SIZE),
        .NUM_ROWS  (NUM_ROWS)
    ) u_left_sw (
        .clk        (clk),
        .rst        (rst),
        .col_in     (left_col),
        .wr_en      (left_lb_wr_en_out & left_lb_valid),
        .window_out (patch_left),
        .valid_out  (left_sw_valid)
    );

    // =========================================================================
    // 2b. Right eye: line_buffer → sliding_window
    // =========================================================================
    wire [NUM_ROWS*DATA_WIDTH-1:0]  right_col;
    wire                             right_lb_valid;
    wire                             right_lb_wr_en_out;
    wire                             right_lb_wr_en_d1_out;

    line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_WIDTH (IMG_WIDTH),
        .NUM_ROWS  (NUM_ROWS)
    ) u_right_lb (
        .clk         (clk),
        .rst         (rst),
        .pixel_in    (pixel_in),
        .wr_en       (right_wr_en),
        .pixel_out   (right_col),
        .valid_out   (right_lb_valid),
        .wr_en_out   (right_lb_wr_en_out),
        .wr_en_d1_out(right_lb_wr_en_d1_out)
    );

    wire [PATCH_W-1:0] patch_right_base;
    wire               right_sw_valid;

    sliding_window #(
        .DATA_WIDTH(DATA_WIDTH),
        .WIN_SIZE  (WIN_SIZE),
        .NUM_ROWS  (NUM_ROWS)
    ) u_right_sw (
        .clk        (clk),
        .rst        (rst),
        .col_in     (right_col),
        // Use wr_en_d1_out (1-cycle delayed) rather than wr_en_out (2-cycle delayed)
        // so the right patch shifts in one cycle BEFORE the delay chain latches it.
        // This ensures delay[0] = {right[c-4]..right[c]} when ST_SNAP fires.
        .wr_en      (right_lb_wr_en_d1_out & right_lb_valid),
        .window_out (patch_right_base),
        .valid_out  (right_sw_valid)
    );

    // =========================================================================
    // 3. Right-eye disparity shift chain
    //    patch_right_delayed[d] = patch_right_base delayed d right-pixel clocks.
    //    Advances on right_lb_wr_en_out (aligned with BRAM read output).
    // =========================================================================
    reg [PATCH_W-1:0] patch_right_delayed [0:MAX_DISP-1];

    always @(posedge clk) begin
        if (rst)
            patch_right_delayed[0] <= 0;
        else if (right_lb_wr_en_out)
            patch_right_delayed[0] <= patch_right_base;
    end

    genvar d;
    generate
        for (d = 1; d < MAX_DISP; d = d + 1) begin : gen_delay
            always @(posedge clk) begin
                if (rst)
                    patch_right_delayed[d] <= 0;
                else if (right_lb_wr_en_out)
                    patch_right_delayed[d] <= patch_right_delayed[d-1];
            end
        end
    endgenerate

    // =========================================================================
    // 4. Time-multiplexed SAD unit + control FSM
    // =========================================================================
    wire both_valid = left_sw_valid & right_sw_valid;

    reg [PATCH_W-1:0]   snap_left;
    reg [PATCH_W-1:0]   snap_right [0:MAX_DISP-1];

    reg [DISP_W-1:0]    cur_disp;
    reg [12:0]          min_cost;
    reg [DISP_W-1:0]    min_disp;

    reg  [PATCH_W-1:0]  sad_patch_l;
    reg  [PATCH_W-1:0]  sad_patch_r;
    reg                  sad_valid_in;
    wire [12:0]          sad_cost_out;
    wire                 sad_valid_out;

    sad_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .WIN_SIZE  (WIN_SIZE)
    ) u_sad (
        .clk        (clk),
        .rst        (rst),
        .patch_left (sad_patch_l),
        .patch_right(sad_patch_r),
        .valid_in   (sad_valid_in),
        .sad_out    (sad_cost_out),
        .valid_out  (sad_valid_out)
    );

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            state        <= ST_IDLE;
            hold_cnt     <= 0;
            cur_disp     <= 0;
            min_cost     <= 13'h1FFF;
            min_disp     <= 0;
            sad_valid_in <= 1'b0;
            sad_patch_l  <= 0;
            sad_patch_r  <= 0;
            disp_out     <= 0;
            disp_valid   <= 1'b0;
            for (i = 1; i < MAX_DISP; i = i + 1)
                snap_right[i] <= 0;
        end else begin
            sad_valid_in <= 1'b0;
            disp_valid   <= 1'b0;

            case (state)

                // ---------------------------------------------------------
                // IDLE: accepting pixels.  Trigger on any left pixel that
                // has valid windows on both eyes.  Deassert ready immediately
                // so no more pixels are accepted while we compute.
                // ---------------------------------------------------------
                ST_IDLE: begin
                    if (left_wr_en & left_sw_valid & right_sw_valid) begin
                        state    <= ST_HOLD;
                        hold_cnt <= 0;
                    end
                end

                // ---------------------------------------------------------
                // HOLD: wait 3 cycles for all pipeline stages to settle.
                //   Cycle 1: BRAM address registered (one cycle after left_wr_en)
                //   Cycle 2: BRAM data registered (patch columns appear on pixel_out)
                //            right_lb_wr_en_d1_out fires → right sw shifts right[c] in
                //   Cycle 3: left sw shifts left[c] in (left_lb_wr_en_out fires)
                //            right_lb_wr_en_out fires  → delay chain latches post-shift
                //            right patch
                // After 3 cycles: patch_left = {left[c-4..c]}, delay[0] = {right[c-4..c]}
                // ---------------------------------------------------------
                ST_HOLD: begin
                    hold_cnt <= hold_cnt + 1;
                    if (hold_cnt == 2'd2)
                        state <= ST_SNAP;
                end

                // ---------------------------------------------------------
                // SNAP: latch patch data and kick off disparity 0.
                // ---------------------------------------------------------
                ST_SNAP: begin
                    snap_left <= patch_left;
                    for (i = 1; i < MAX_DISP; i = i + 1)
                        snap_right[i] <= patch_right_delayed[i];

                    cur_disp  <= 0;
                    min_cost  <= 13'h1FFF;
                    min_disp  <= 0;

                    sad_patch_l  <= patch_left;
                    sad_patch_r  <= patch_right_delayed[0];
                    sad_valid_in <= 1'b1;

                    state <= ST_RUNNING;
                end

                // ---------------------------------------------------------
                // RUNNING: iterate disparities; emit result when done.
                // ---------------------------------------------------------
                ST_RUNNING: begin
                    if (sad_valid_out) begin
                        if (sad_cost_out < min_cost) begin
                            min_cost <= sad_cost_out;
                            min_disp <= cur_disp;
                        end

                        if (cur_disp == MAX_DISP - 1) begin
                            disp_out   <= (sad_cost_out < min_cost)
                                          ? cur_disp : min_disp;
                            disp_valid <= 1'b1;
                            state      <= ST_IDLE;
                        end else begin
                            cur_disp     <= cur_disp + 1;
                            sad_patch_l  <= snap_left;
                            sad_patch_r  <= snap_right[cur_disp + 1];
                            sad_valid_in <= 1'b1;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
