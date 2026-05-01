# FPGA Board Note

This note records the current target FPGA board and the immediate deployment
implications for the stereo-depth model in `v2/`.

## FPGA Device

- AMD Artix 7 `XC7A200T-1SBG484C`
- 33,650 logic slices
- 4 LUTs and 8 flip-flops per slice
- 13 Mbits block RAM
- 10 clock management tiles with PLLs
- 740 DSP slices
- Internal clock speeds exceeding 450 MHz
- XADC on chip
- Up to 3.75 Gbps GTP transceivers

## Board Features

- Programmable over JTAG, Quad-SPI Flash, microSD, and USB flash drive
- 512 MB DDR3 at 400 MHz (800 MT/s)
- 32 MB Quad-SPI flash
- microSD slot for nonvolatile storage
- Dedicated USB port for JTAG programming and data transfer
- 12 V power input
- HDMI sink and HDMI source
- Mini DisplayPort source
- 24-bit audio codec with four 3.5 mm jacks
- Gigabit Ethernet PHY
- User EEPROM
- USB-UART bridge
- Digilent parallel transfer interface (DTPI)
- Digilent serial peripheral interface (DSPI)
- 128x32 monochrome OLED
- 5 user push buttons
- 8 user switches
- 8 user LEDs
- USB HID host
- 160-pin FMC LPC connector
- Four Pmod ports
- XADC-accessible Pmod signals

## Current Model Size

Current best larger-input model:

- Full stereo model parameters: `65,556`
- Feature-extractor sub-model parameters: `8,052`

Approximate weight storage only:

- Full model at FP32: `2,097,792` bits, about `256 KB`
- Full model at INT8: `524,448` bits, about `64 KB`
- Feature extractor at FP32: `257,664` bits, about `31.5 KB`
- Feature extractor at INT8: `64,416` bits, about `7.9 KB`

## Fit Assessment

Weight storage by itself is not the main problem.

The full model's weights fit inside the board's 13 Mbits of BRAM even at FP32,
and very comfortably at INT8. The problem is the full-frame working set and the
cost-volume part of the network.

For the current `128 x 384 x 64` cost volume:

- elements: `128 * 384 * 64 = 3,145,728`
- if stored at INT8: about `25.2 Mbits`
- if stored at FP16: about `50.3 Mbits`

That means the full model is not a practical direct fit as one monolithic HLS
block on this Artix-7 device. This matches the existing code in
`v2/hls4ml/convert_student.py`, which explicitly exports only the shared feature
extractor for FPGA deployment and leaves disparity matching to a separate
hardware correlator.

## Practical Conclusion

- The full Keras stereo model should be treated as a training/reference model.
- The FPGA target should remain the feature extractor plus a custom matching
  stage, not the full end-to-end network.
- Quantization is still needed later, especially for activation storage,
  bandwidth, and DSP pressure.
- INT8 or mixed fixed-point is the most plausible next step for deployment.

## Recommended Deployment Path

1. Quantize the exported feature extractor.
2. Keep the disparity or cost-volume stage as a custom streaming hardware block.
3. Avoid materializing the full cost volume in BRAM.
4. Reuse DSPs and line buffers instead of mapping the full model in parallel.