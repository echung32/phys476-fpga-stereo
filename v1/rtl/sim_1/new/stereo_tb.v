// =============================================================================
// stereo_tb.v
// =============================================================================
// Testbench for the stereo depth pipeline.
//
// Pixel stream uses a valid/ready handshake:
//   - pix_valid is held high while the testbench has a pixel to send.
//   - A pixel is transferred on the rising edge where BOTH pix_valid and
//     ready are high.
//   - The testbench advances to the next pixel after each accepted transfer.
//
// STREAM ORDER: INTERLEAVED.
//   For each row r, for each column c: right[r][c] then left[r][c].
//   This aligns the right-eye delay chain so that at left column c,
//   delay[d] holds the right-eye patch from column c-d.
// =============================================================================

`timescale 1ns / 1ps

module stereo_tb;

    // -------------------------------------------------------------------------
    // Parameters (must match DUT)
    // -------------------------------------------------------------------------
    parameter DATA_WIDTH = 8;
    parameter IMG_WIDTH  = 512;
    parameter IMG_HEIGHT = 512;
    parameter WIN_SIZE   = 5;
    parameter NUM_ROWS   = 5;
    parameter MAX_DISP   = 16;
    parameter DISP_W     = 4;   // $clog2(MAX_DISP)

    parameter CLK_PERIOD = 10;  // ns  (100 MHz)

    // -------------------------------------------------------------------------
    // Clock and reset
    // -------------------------------------------------------------------------
    reg clk = 0;
    reg rst = 1;

    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------------------------
    reg  [DATA_WIDTH-1:0] pixel_in;
    reg                   pix_valid;
    wire                  ready;
    wire [DISP_W-1:0]     disp_out;
    wire                  disp_valid;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    stereo_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_WIDTH (IMG_WIDTH),
        .WIN_SIZE  (WIN_SIZE),
        .NUM_ROWS  (NUM_ROWS),
        .MAX_DISP  (MAX_DISP)
    ) dut (
        .clk        (clk),
        .rst        (rst),
        .pixel_in   (pixel_in),
        .pix_valid  (pix_valid),
        .ready      (ready),
        .disp_out   (disp_out),
        .disp_valid (disp_valid)
    );

    // -------------------------------------------------------------------------
    // Image memories
    // -------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] left_img  [0:IMG_WIDTH*IMG_HEIGHT-1];
    reg [DATA_WIDTH-1:0] right_img [0:IMG_WIDTH*IMG_HEIGHT-1];

    // -------------------------------------------------------------------------
    // Output file
    // -------------------------------------------------------------------------
    integer out_fd;
    integer out_pixel_count;

    // -------------------------------------------------------------------------
    // Capture: write each valid disparity pixel to file
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (disp_valid) begin
            $fwrite(out_fd, "%02x\n", disp_out);
            out_pixel_count = out_pixel_count + 1;
        end
    end

    // -------------------------------------------------------------------------
    // Task: send one pixel using valid/ready handshake
    // -------------------------------------------------------------------------
    task send_pixel;
        input [DATA_WIDTH-1:0] px;
        begin
            pixel_in  = px;
            pix_valid = 1'b1;
            // Wait for a rising edge where ready is also high
            @(posedge clk);
            while (!ready) @(posedge clk);
            #1;  // Small skew past the clock edge before changing inputs
        end
    endtask

    // -------------------------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------------------------
    integer row, col, flat_idx;
    integer r;

    initial begin
        // --- Load images ---------------------------------------------------
        $display("[TB] Loading left_img.hex ...");
        $readmemh("../../data/left_img.hex", left_img);
        $display("[TB] Loading right_img.hex ...");
        $readmemh("../../data/right_img.hex", right_img);

        // --- Open output file ----------------------------------------------
        out_fd = $fopen("../../output/depth_out.hex", "w");
        if (out_fd == 0) begin
            $display("[TB] ERROR: Cannot open depth_out.hex for writing.");
            $finish;
        end
        out_pixel_count = 0;

        // --- Reset sequence ------------------------------------------------
        pixel_in  = 0;
        pix_valid = 0;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;

        $display("[TB] Starting pixel stream ...");

        // --- Stream pixels using valid/ready handshake ---------------------
        // INTERLEAVED layout: for each column c, right[r][c] then left[r][c].
        // This ensures that when the FSM snapshots at left column c, the
        // right-eye delay chain holds patches from right cols c, c-1, ..., c-d.
        // (The previous half-row layout sent all 512 right pixels before any
        //  left pixel, so the delay chain was always misaligned by 512 columns.)
        for (row = 0; row < IMG_HEIGHT; row = row + 1) begin
            for (col = 0; col < IMG_WIDTH; col = col + 1) begin
                flat_idx = row * IMG_WIDTH + col;
                send_pixel(right_img[flat_idx]);  // right first
                send_pixel(left_img[flat_idx]);   // then left
            end
        end

        // --- De-assert, drain pipeline  (allow last SAD to complete) -------
        pix_valid = 0;
        pixel_in  = 0;
        // Worst-case pipeline drain: MAX_DISP * 26 + a few overhead cycles
        repeat (500) @(posedge clk);

        // --- Wrap up -------------------------------------------------------
        $display("[TB] Stream complete. Output pixels written: %0d", out_pixel_count);
        $display("[TB] Expected approx %0d pixels.",
                 (IMG_HEIGHT - NUM_ROWS + 1) * (IMG_WIDTH - WIN_SIZE + 1 - MAX_DISP));
        $fclose(out_fd);
        $display("[TB] Simulation finished.");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Timeout watchdog.
    // With interleaved stream + backpressure + ST_HOLD:
    //   each valid pixel pair takes ~420 stall cycles + 2 input cycles.
    //   Valid output positions: ~249936; total cycles ~105M.
    //   Plus: non-valid positions (row<4, col<19) also pass through quickly.
    // Budget: 1600 * 1,000,000 ns = 1.6 billion ns.
    // -------------------------------------------------------------------------
    initial begin
        // 1600 iterations of 1,000,000 ns = 1.6 billion ns
        for (r = 0; r < 1600; r = r + 1)
            #1000000;
        $display("[TB] TIMEOUT — simulation took too long.");
        $fclose(out_fd);
        $finish;
    end

    // -------------------------------------------------------------------------
    // Optional: VCD waveform dump (uncomment to enable)
    // -------------------------------------------------------------------------
    // initial begin
    //     $dumpfile("stereo_sim.vcd");
    //     $dumpvars(0, stereo_tb);
    // end

endmodule
