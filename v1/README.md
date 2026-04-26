# v1 Baseline Track

`v1` is the preserved reference implementation for the original stereo depth pipeline.

The baseline now lives entirely under `v1/`:

- RTL: `rtl/sources_1/new/`
- Testbench: `rtl/sim_1/new/`
- Inputs: `data/`
- Outputs: `output/`
- Visualizer: `plot_depth.py`

## Baseline capabilities

- Interleaved grayscale stereo pixel stream input
- 5x5 local window generation
- 16-candidate disparity search
- `.hex`-based simulation and evaluation path
- Python visualization via `uv run python v1/plot_depth.py`

## Migration boundary

The matching core is the clean replacement boundary. Line buffering, sliding-window generation, testbench structure, and `.hex` evaluation remain the reference interface that `v2` is designed around.