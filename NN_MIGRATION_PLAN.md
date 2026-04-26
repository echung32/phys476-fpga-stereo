# Neural Stereo Migration Plan

## Goal

Migrate the current stereo depth project from a naive local SAD-based matcher to a neural-network-based method that can later target hls4ml, while preserving the existing Verilog pipeline as a reference implementation.

This plan assumes:

- development will continue on a separate Ubuntu system with NVIDIA GPUs
- that Ubuntu system will not have Vivado or Vitis HLS installed initially
- hls4ml work will focus on model conversion, software compilation, and interface preparation, not FPGA synthesis

## Current Project Summary

The current design is a simulation-only stereo depth pipeline implemented in Verilog.

- Input: 512x512 grayscale stereo frames as an interleaved pixel stream
- Matching method: 5x5 local SAD over 16 disparity candidates
- Existing reusable components:
  - line buffering
  - sliding-window patch generation
  - simulation/testbench flow
  - .hex-based evaluation and visualization

The clean migration boundary is the matching core. The patch-generation and evaluation flow should remain available as the baseline.

## High-Level Strategy

The project should be split into two tracks.

### v1

Keep the current design as the baseline reference.

- Preserve current Verilog sources and testbench behavior
- Preserve current .hex input and output workflow
- Preserve current Python visualization and metric path
- Use v1 as the correctness and comparison target

### v2

Create a neural stereo pipeline intended for future hls4ml deployment.

- Replace SAD scoring with a learned model
- Keep the hardware-facing interface as close as possible to the existing 5x5 patch plus disparity-candidate structure
- Validate the model in software first
- Convert with hls4ml later on Ubuntu, even before Vivado is available

## Recommended Model Architecture

The deployable model should not be a full-image stereo foundation model.

Instead, use a two-tier approach:

1. Teacher model

- Use a strong pretrained stereo model offline for benchmarking and pseudo-label generation
- Candidate teacher: FoundationStereo
- Role: produce dense disparity maps, generate pseudo-labels, and help domain adaptation
- This model is not intended for hls4ml export

2. Student model

- Train a compact patch-based model that is realistic for hls4ml conversion
- Input should match the eventual hardware-facing interface as much as possible
- Preferred model types:
  - small 2D CNN over stacked left/right patches
  - compact MLP over flattened or engineered patch features
- Output should be one of:
  - a matching cost for a single disparity candidate
  - a binary or probabilistic match score for a single candidate

The most practical v2 path is:

- keep the current 16-disparity search loop conceptually
- evaluate one candidate disparity at a time with the student model
- choose the minimum cost or highest match score across the 16 candidates

This keeps the software path close to the eventual HDL integration path.

## Why Not Whole-Image Zero-Shot Deployment

Whole-image stereo models are useful as teachers or evaluation baselines, but they are not the right deployment target here.

Reasons:

- they are too large for a practical hls4ml deployment path in this project
- they depend on GPU-oriented architectures and operators that are poor FPGA migration targets
- they do not align cleanly with the current patch-based Verilog dataflow

Zero-shot whole-image stereo should therefore remain an offline tool, not the final target architecture.

## Dataset Plan

Use datasets in stages rather than downloading everything immediately.

### Stage 1: small bring-up dataset

Use a very small subset first to validate the training and preprocessing pipeline.

- synthetic sample data from Scene Flow if convenient
- current project synthetic data for quick sanity tests

### Stage 2: supervised pretraining

Primary recommendation: Scene Flow subsets.

Why:

- dense stereo disparity labels
- standard stereo pretraining source
- good for learning local matching behavior

Constraint:

- full Scene Flow is very large, so begin with a controlled subset

### Stage 3: real-domain fine-tuning

Primary recommendation: KITTI 2015 stereo training split.

Why:

- real driving scenes
- standard stereo benchmark
- useful for adapting away from purely synthetic data

### Stage 4: cross-domain evaluation

Use Middlebury as a compact spot-check dataset.

Why:

- good ground truth quality
- useful for checking generalization beyond KITTI-like scenes

### Optional Stage 5: teacher-generated pseudo-labels

If you have your own stereo image pairs or additional real data, run a strong teacher model to generate pseudo-labels, then fine-tune the student model on confidence-filtered labels.

## Environment Plan

The environment should be separated by purpose.

### Windows local system

Keep Windows for lightweight project editing and existing repo management if needed, but do not make it the main hls4ml target environment.

### Ubuntu GPU system

This should become the primary v2 development environment.

Recommended responsibilities:

- dataset download and storage
- preprocessing and patch extraction
- teacher-model inference
- student-model training
- hls4ml conversion and software compile checks

### Vivado constraint

Because Vivado is not available, do not plan around synthesis or timing results yet.

What is still useful without Vivado:

- train the student model
- export and convert models with hls4ml
- run hls4ml compile and prediction checks where supported
- compare native model predictions against hls4ml predictions
- prepare the interface and integration design for later FPGA insertion

## Proposed Repository Layout

Suggested future layout:

```text
project_root/
├── v1/
│   ├── rtl/
│   ├── sim/
│   ├── data/
│   ├── output/
│   └── docs/
├── v2/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── manifests/
│   ├── models/
│   │   ├── teacher/
│   │   └── student/
│   ├── training/
│   ├── evaluation/
│   ├── hls4ml/
│   ├── exports/
│   └── docs/
├── v1/
│   ├── plot_depth.py
│   ├── data/
│   ├── output/
│   └── rtl/
├── REPORT.md
└── NN_MIGRATION_PLAN.md
```

This split keeps the current implementation intact while letting v2 evolve independently.

## v2 Software Pipeline

The v2 implementation should be built in this order.

### Phase 1: baseline preservation

- freeze the current implementation as v1
- keep the current report and simulation flow accessible
- confirm the existing visualization flow still works

### Phase 2: data ingestion

- add dataset loaders for Scene Flow, KITTI, and Middlebury
- normalize the data into a consistent internal format
- create manifest files for train, validation, and test splits

### Phase 3: patch extraction

- extract left 5x5 patches centered on valid pixels
- build corresponding right-side candidate patches for disparities 0 through 15
- create labels from ground-truth disparity

Two practical labeling options:

- binary match labels per candidate disparity
- scalar cost or ranking labels centered on the true disparity

### Phase 4: student baseline

- train a minimal student model first
- keep the architecture deliberately simple and hls4ml-friendly
- establish a software-only baseline before adding distillation

### Phase 5: teacher integration

- run FoundationStereo or another strong stereo teacher offline
- generate dense disparity maps on real or custom stereo inputs
- filter low-confidence teacher outputs if confidence information is available
- use these pseudo-labels to fine-tune the student

### Phase 6: full disparity reconstruction

- for each pixel, evaluate the student across the 16 disparity candidates
- select the best candidate with argmin or argmax depending on scoring convention
- reconstruct the full disparity map
- compare against ground truth and against v1 outputs where useful

### Phase 7: hls4ml conversion

- export the student model in a supported form
- convert it with hls4ml on Ubuntu
- run software compile and prediction checks
- measure prediction parity between the native model and the hls4ml model

### Phase 8: hardware-interface preparation

- document how the student model input maps onto the existing Verilog patch interface
- define the future replacement boundary around the current SAD unit and control logic
- postpone HDL edits until the model and conversion path are stable

## hls4ml Design Constraints

The student model should be designed for hls4ml compatibility from the start.

Practical constraints:

- prefer standard Keras-supported layers such as Dense, Conv2D, Pooling, Reshape, Concatenate, and common activations
- avoid Lambda layers
- avoid arbitrary operators where equivalent Keras layers exist
- keep tensor layouts simple
- keep the model small enough that future quantization and FPGA mapping remain realistic

The first student should prioritize convertibility and clarity over peak accuracy.

## Validation Plan

Validation should happen in successive layers.

### Validation 1: dataset correctness

- verify patch extraction aligns with ground-truth disparity
- confirm candidate windows are generated correctly for the 0 to 15 disparity range

### Validation 2: native model behavior

- train on a small subset first
- reconstruct disparity maps in software
- measure simple metrics such as RMSE, bad-pixel rate, or accuracy within 1 pixel

### Validation 3: teacher-student benefit

- compare supervised-only student against teacher-assisted fine-tuning
- confirm whether the teacher improves real-domain performance

### Validation 4: hls4ml parity

- compare native framework predictions against hls4ml predictions on held-out patch batches
- confirm that conversion preserves acceptable numeric behavior

### Validation 5: interface readiness

- confirm that the deployable student can be expressed using the current patch-based interface assumptions
- document future HDL insertion points

## Recommended Immediate Next Steps

When you move to Ubuntu, execute this order:

1. Create the v1 and v2 directory split.
2. Set up the Ubuntu Python environment for training and preprocessing.
3. Add dataset loaders and patch-extraction scripts.
4. Train a tiny student baseline on a small synthetic subset.
5. Reconstruct full disparity maps in software over the 16-candidate search range.
6. Add teacher-based pseudo-label experiments using FoundationStereo if GPU time is available.
7. Convert the student with hls4ml and run software parity checks.

## Deliverables To Aim For Before Any Vivado Work

The first milestone should produce:

- a preserved v1 baseline
- a working v2 data pipeline
- a trainable compact student model
- software disparity-map reconstruction using the student
- one or more evaluation reports on held-out data
- an hls4ml-converted model that can be validated in software
- a written interface plan for future HDL integration

## Out Of Scope For Now

The following should be deferred until later:

- Vivado synthesis
- Vitis HLS synthesis reports
- FPGA timing closure
- resource optimization against a specific board target
- final RTL replacement of the current SAD path

## Summary

The correct migration path is not to replace the current pipeline with a full-image foundation stereo model. The practical path is to use a strong stereo model offline as a teacher, then train a small hls4ml-compatible patch-based student that preserves the current disparity-search structure.

That gives you:

- a clean comparison against the existing v1 design
- a realistic future FPGA deployment path
- useful work that can be completed on Ubuntu with GPUs even without Vivado