`timescale 1ns/1ps
// myproject_tb.sv — Behavioral testbench for hls4ml feature extractor
//
// Supports multiple input frames via a manifest file.
// Each line of tb_data/frame_manifest.txt names one .mem file (absolute or
// relative to the xsim run directory) to simulate in sequence.
// If the manifest is absent, falls back to a single tb_data/tb_input_hex.mem.
//
// Outputs all 16 feature channels for every pixel to:
//   tb_data/rtl_output_<frame_stem>.hex  — 46080 lines of 256-bit hex
//   tb_data/rtl_output_<frame_stem>.f32  — 46080×16 floats as space-separated
//                                          fixed<16,6> decoded values
//
// Use with Vivado xsim via vivado_sim.tcl.

module myproject_tb;

  // -------------------------------------------------------------------------
  // Clock and reset
  // -------------------------------------------------------------------------
  logic clk   = 1'b0;
  logic rst_n = 1'b0;

  // 100 MHz clock (10 ns period)
  always #5 clk = ~clk;

  // -------------------------------------------------------------------------
  // DUT interface signals
  // -------------------------------------------------------------------------
  logic [15:0]  in_tdata  = '0;
  logic         in_tvalid = 1'b0;
  logic         in_tready;

  logic [255:0] out_tdata;
  logic         out_tvalid;
  logic         out_tready = 1'b1;

  logic ap_start = 1'b0;
  logic ap_done;
  logic ap_ready;
  logic ap_idle;

  // -------------------------------------------------------------------------
  // DUT instantiation
  // -------------------------------------------------------------------------
  myproject dut (
    .image_r_TDATA     (in_tdata),
    .image_r_TVALID    (in_tvalid),
    .image_r_TREADY    (in_tready),
    .layer7_out_TDATA  (out_tdata),
    .layer7_out_TVALID (out_tvalid),
    .layer7_out_TREADY (out_tready),
    .ap_clk            (clk),
    .ap_rst_n          (rst_n),
    .ap_start          (ap_start),
    .ap_done           (ap_done),
    .ap_ready          (ap_ready),
    .ap_idle           (ap_idle)
  );

  // -------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------
  localparam int H     = 160;
  localparam int W     = 288;
  localparam int N_PIX = H * W;   // 46 080
  localparam int N_CH  = 16;

  // -------------------------------------------------------------------------
  // Test data storage
  // -------------------------------------------------------------------------
  logic [15:0]  input_pixels  [0:N_PIX-1];
  logic [255:0] output_beats  [0:N_PIX-1];  // full 256-bit (16 ch) per pixel pos

  // -------------------------------------------------------------------------
  // Task: run one inference from a given .mem file, write output files
  // -------------------------------------------------------------------------
  integer fout_hex, fout_f32;
  integer beat, ob;
  real    fval;

  task run_frame(input string mem_path, input string out_stem);
    integer fd_h, fd_f;
    string  hex_path, f32_path;
    integer out_cnt;
    integer ch;

    // Load input pixels
    $readmemh(mem_path, input_pixels);
    $display("INFO [%0t] Starting frame: %s", $time, mem_path);

    // Reset DUT between frames (5 cycles)
    rst_n = 1'b0;
    repeat (5) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    // ap_start pulse
    @(negedge clk);
    ap_start = 1'b1;
    @(posedge clk); #1;
    ap_start = 1'b0;

    // Fork: stream input and capture output in parallel
    out_cnt = 0;
    fork
      // --- Input driver ---
      begin
        for (beat = 0; beat < N_PIX; beat++) begin
          @(negedge clk);
          in_tdata  = input_pixels[beat];
          in_tvalid = 1'b1;
          @(posedge clk);
          while (!in_tready) @(posedge clk);
          #1;
        end
        @(negedge clk);
        in_tvalid = 1'b0;
        in_tdata  = '0;
      end
      // --- Output capture ---
      begin
        while (out_cnt < N_PIX) begin
          @(posedge clk);
          if (out_tvalid && out_tready) begin
            output_beats[out_cnt] = out_tdata;
            out_cnt++;
          end
        end
        $display("INFO [%0t] All %0d output beats captured", $time, N_PIX);
      end
    join

    // All N_PIX output beats captured — inference is complete.
    // ap_done fires as a 1-cycle pulse around the same time as the last
    // output beat, so it may have already come and gone during the fork.
    // Wait a few extra cycles to let it settle, then proceed.
    repeat (20) @(posedge clk);
    $display("INFO [%0t] Frame complete (all beats captured): %s", $time, out_stem);

    // Write hex output (one 256-bit word per line)
    hex_path = {"tb_data/rtl_output_", out_stem, ".hex"};
    fd_h = $fopen(hex_path, "w");
    for (ob = 0; ob < N_PIX; ob++)
      $fwrite(fd_h, "%064x\n", output_beats[ob]);
    $fclose(fd_h);
    $display("INFO: Wrote %s", hex_path);

    // Write decoded float output (fixed<16,6> → real, space-separated per pixel)
    f32_path = {"tb_data/rtl_output_", out_stem, ".f32"};
    fd_f = $fopen(f32_path, "w");
    for (ob = 0; ob < N_PIX; ob++) begin
      for (ch = 0; ch < N_CH; ch++) begin
        fval = $signed(output_beats[ob][ch*16 +: 16]) / 64.0;
        if (ch < N_CH-1)
          $fwrite(fd_f, "%.6f ", fval);
        else
          $fwrite(fd_f, "%.6f\n", fval);
      end
    end
    $fclose(fd_f);
    $display("INFO: Wrote %s", f32_path);

  endtask

  // -------------------------------------------------------------------------
  // Main: initial reset, then iterate over manifest
  // -------------------------------------------------------------------------
  integer manifest_fd;
  string  mem_line, stem;
  integer n_frames, timeout_flag;

  initial begin
    timeout_flag = 0;

    // Initial reset
    rst_n = 1'b0;
    repeat (10) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    repeat (5)  @(posedge clk);

    // Try to open manifest; fall back to single frame
    manifest_fd = $fopen("tb_data/frame_manifest.txt", "r");
    n_frames = 0;

    if (manifest_fd != 0) begin
      $display("INFO: Reading frame manifest tb_data/frame_manifest.txt");
      while (!$feof(manifest_fd)) begin
        if ($fgets(mem_line, manifest_fd)) begin
          // Strip trailing newline
          mem_line = mem_line.substr(0, mem_line.len()-2);
          if (mem_line.len() > 0) begin
            // Derive stem: strip directory and .mem extension
            stem = mem_line;
            // Remove path prefix up to last /
            begin
              automatic int last_slash = -1;
              for (int ci = 0; ci < stem.len(); ci++)
                if (stem[ci] == "/") last_slash = ci;
              if (last_slash >= 0)
                stem = stem.substr(last_slash+1, stem.len()-1);
            end
            // Remove .mem suffix
            if (stem.len() > 4 && stem.substr(stem.len()-4, stem.len()-1) == ".mem")
              stem = stem.substr(0, stem.len()-5);
            run_frame(mem_line, stem);
            n_frames++;
          end
        end
      end
      $fclose(manifest_fd);
      $display("INFO: Processed %0d frame(s) from manifest", n_frames);
    end else begin
      $display("INFO: No manifest found, running single frame tb_data/tb_input_hex.mem");
      run_frame("tb_data/tb_input_hex.mem", "frame0");
      n_frames = 1;
    end

    timeout_flag = 1;
    $display("INFO: All %0d frame(s) complete", n_frames);
    $finish;
  end

  // -------------------------------------------------------------------------
  // Timeout watchdog — 20 ms per frame, up to 20 frames = 400 ms max
  // -------------------------------------------------------------------------
  initial begin
    #400_000_000;  // 400 ms at 1 ns resolution
    if (!timeout_flag) begin
      $display("ERROR [%0t] Simulation timeout", $time);
      $finish;
    end
  end

endmodule
