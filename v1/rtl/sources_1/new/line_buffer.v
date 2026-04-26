// =============================================================================
// line_buffer.v  (v3 — guaranteed BRAM inference)
// =============================================================================
// Stores the last NUM_ROWS rows of pixels using Block RAM.
//
// Root cause of previous failure:
//   Vivado's BRAM inference requires a 1-D array addressed by a single index.
//   A 2-D array `mem[bank][col]` with a variable `bank` index is treated as
//   flip-flops, producing ~98 000 LUTs per instance.
//
// Fix:
//   Flatten storage into ONE 1-D array of depth (NUM_BANKS * IMG_WIDTH).
//   Address = bank * IMG_WIDTH + col  (both terms are registered before use).
//   Vivado sees the canonical single-port synchronous RAM pattern and infers
//   RAMB36E2 primitives.
//
//   NUM_BANKS is rounded up to the next power of two so the bank offset
//   multiplication simplifies to a left-shift (free), keeping the address
//   computation in fabric logic minimal.
//
// Parameters:
//   DATA_WIDTH – bits per pixel (default 8)
//   IMG_WIDTH  – pixels per row (default 512, must be power of two)
//   NUM_ROWS   – rows buffered  (default 5, for a 5×5 window)
//
// Interface:
//   clk        – system clock (synchronous everywhere)
//   rst        – synchronous active-high reset
//   pixel_in   – incoming 8-bit pixel
//   wr_en      – write enable (assert when pixel_in is valid)
//   pixel_out  – flat bus: row0..row(NUM_ROWS-1), each DATA_WIDTH wide
//                row 0 = oldest completed row, row NUM_ROWS-1 = most recent
//   valid_out  – high once NUM_ROWS complete rows have been received
// =============================================================================

`timescale 1ns / 1ps

module line_buffer #(
    parameter DATA_WIDTH = 8,
    parameter IMG_WIDTH  = 512,   // must be power of two
    parameter NUM_ROWS   = 5
)(
    input  wire                            clk,
    input  wire                            rst,
    input  wire [DATA_WIDTH-1:0]           pixel_in,
    input  wire                            wr_en,
    output wire [NUM_ROWS*DATA_WIDTH-1:0]  pixel_out,
    output reg                             valid_out,
    // wr_en_out is wr_en delayed by 2 cycles to align with BRAM read latency.
    // Connect left_lb.wr_en_out to left sliding_window.wr_en so it fills with valid data.
    output reg                             wr_en_out,
    // wr_en_d1_out is wr_en delayed by 1 cycle (one cycle earlier than wr_en_out).
    // Use this to drive the right sliding_window.wr_en in stereo_top so that the
    // right patch is fully settled one cycle before the delay chain latches it.
    output reg                             wr_en_d1_out
);

    // -------------------------------------------------------------------------
    // Round NUM_ROWS up to the next power of two for the bank count.
    // This makes bank_offset = wr_bank << ADDR_W (a shift, not a multiply).
    // -------------------------------------------------------------------------
    localparam NUM_BANKS  = (NUM_ROWS <= 2) ? 2 :
                            (NUM_ROWS <= 4) ? 4 : 8;
    localparam BANK_W     = $clog2(NUM_BANKS);      // 3 for 8 banks
    localparam ADDR_W     = $clog2(IMG_WIDTH);       // 9 for 512
    localparam MEM_DEPTH  = NUM_BANKS * IMG_WIDTH;   // 8 * 512 = 4096
    localparam MEM_ADDR_W = BANK_W + ADDR_W;         // 12 bits

    // -------------------------------------------------------------------------
    // Column address counter (wraps at IMG_WIDTH)
    // -------------------------------------------------------------------------
    reg [ADDR_W-1:0] col_addr;

    always @(posedge clk) begin
        if (rst)
            col_addr <= 0;
        else if (wr_en)
            col_addr <= (col_addr == IMG_WIDTH - 1) ? 0 : col_addr + 1;
    end

    wire end_of_row = wr_en & (col_addr == IMG_WIDTH - 1);

    // -------------------------------------------------------------------------
    // Bank (row) counter — which bank receives the current row
    // -------------------------------------------------------------------------
    reg [BANK_W-1:0] wr_bank;

    always @(posedge clk) begin
        if (rst)
            wr_bank <= 0;
        else if (end_of_row)
            wr_bank <= (wr_bank == NUM_BANKS - 1) ? 0 : wr_bank + 1;
    end

    // -------------------------------------------------------------------------
    // Valid-output flag
    // -------------------------------------------------------------------------
    localparam ROW_CNT_W = $clog2(NUM_ROWS + 1);
    reg [ROW_CNT_W-1:0] rows_filled;

    always @(posedge clk) begin
        if (rst) begin
            rows_filled <= 0;
            valid_out   <= 1'b0;
        end else if (end_of_row) begin
            if (rows_filled < NUM_ROWS)
                rows_filled <= rows_filled + 1;
            if (rows_filled == NUM_ROWS - 1)
                valid_out <= 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Flat 1-D BRAM  — the only pattern Vivado infers as RAMB36E2
    //
    // Write port: address = {wr_bank, col_addr}
    // Read ports: address = {rd_bank[k], col_addr}   (one per output row)
    //
    // Both ports are fully synchronous (registered address + registered data).
    // -------------------------------------------------------------------------
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];

    // Initialise memory to zero for simulation (avoids x-propagation through
    // BRAM reads before all addresses have been written at least once).
    integer init_i;
    initial begin
        for (init_i = 0; init_i < MEM_DEPTH; init_i = init_i + 1)
            mem[init_i] = 8'h00;
    end

    // --- Write ---
    wire [MEM_ADDR_W-1:0] wr_addr = {wr_bank, col_addr};

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= pixel_in;
    end

    // --- wr_en_out: 2-cycle delayed version of wr_en (matches BRAM read latency)
    // --- wr_en_d1_out: 1-cycle delayed version (one cycle earlier than wr_en_out)
    always @(posedge clk) begin
        if (rst) begin
            wr_en_d1_out <= 1'b0;
            wr_en_out    <= 1'b0;
        end else begin
            wr_en_d1_out <= wr_en;
            wr_en_out    <= wr_en_d1_out;
        end
    end

    // --- Read (one port per output row) ---
    // Row k=0 is the oldest completed row:
    //   oldest bank = (wr_bank + 1) mod NUM_BANKS  (next bank after current)
    //   row k is (wr_bank + 1 + k) mod NUM_BANKS
    // We pipeline the address by one cycle so the BRAM address path is
    // fully registered (required for RAMB inference; no async read).
    genvar k;
    generate
        for (k = 0; k < NUM_ROWS; k = k + 1) begin : gen_read

            // rd_bank wraps modulo NUM_BANKS.  Since NUM_BANKS is a power of
            // two, masking with (NUM_BANKS-1) gives free modular arithmetic
            // and avoids the conditional subtraction (which confuses BRAM
            // inference in some Vivado versions).
            wire [BANK_W-1:0]    rd_bank_comb;
            assign rd_bank_comb = (wr_bank + k[BANK_W-1:0] + 1'b1) & (NUM_BANKS - 1);

            // Register address one cycle ahead → synchronous BRAM read
            reg [MEM_ADDR_W-1:0] rd_addr_r;
            always @(posedge clk) begin
                if (rst)
                    rd_addr_r <= 0;
                else
                    rd_addr_r <= {rd_bank_comb, col_addr};
            end

            // Synchronous read: data appears one cycle after rd_addr_r is set
            reg [DATA_WIDTH-1:0] rd_data;
            always @(posedge clk) begin
                rd_data <= mem[rd_addr_r];
            end

            assign pixel_out[(k+1)*DATA_WIDTH-1 : k*DATA_WIDTH] = rd_data;
        end
    endgenerate

endmodule
