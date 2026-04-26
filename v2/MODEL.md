# v2 Model And Training Flow

## Overview

`v2` replaces the synthetic `.hex`-driven matcher with a real-data stereo pipeline that still preserves the deployable hardware assumption:

- a left 5x5 patch
- one right 5x5 candidate patch
- a fixed candidate disparity range of `0..15`
- one scalar match score per candidate

That means the software model is not a full-image stereo network. The full disparity map is reconstructed by evaluating the student model repeatedly over all candidate disparities for each valid pixel.

## Models

### Student model

The deployable model is the native Keras student in `v2/models/keras_student.py`.

Input:

- shape: `(5, 5, 2)`
- channel 0: grayscale left patch
- channel 1: grayscale right patch for one disparity candidate

Architecture:

1. `Conv2D(8, 3x3, padding="same", relu)`
2. `Conv2D(8, 3x3, padding="valid", relu)`
3. `Flatten`
4. `Dense(32, relu)`
5. `Dense(1, sigmoid)`

Why this model:

- it is small enough to stay realistic for hls4ml conversion
- it uses standard Keras layers directly supported by hls4ml
- it avoids `Lambda` layers and unsupported raw operators
- it keeps the input contract aligned with the eventual FPGA-facing patch interface

Output semantics:

- the model returns a scalar match probability for a single candidate disparity
- higher is better
- final disparity is chosen with `argmax` across the 16 candidate scores

### Teacher model

The currently implemented teacher is a larger patch-based Keras model used for distillation.

Architecture:

1. `Conv2D(32, 3x3, padding="same", relu)`
2. `Conv2D(32, 3x3, padding="same", relu)`
3. `Conv2D(16, 3x3, padding="valid", relu)`
4. `Flatten`
5. `Dense(128, relu)`
6. `Dense(64, relu)`
7. `Dense(1, sigmoid)`

This teacher is still not the deployable model. Its purpose is:

- stronger supervision over the same patch-matching problem
- teacher-student distillation experiments
- improving the final student without changing the hardware-facing interface

This is a practical intermediate step before adding a much larger full-image stereo teacher such as FoundationStereo.

## Datasets

### KITTI fine-tuning and evaluation

The currently validated real-data source is `UniflexAI/mini_kitti` on Hugging Face.

Schema used by the loader:

- `left_image`: decoded RGB image
- `right_image`: decoded RGB image
- `disparity`: zipped `.npy` payload containing a float32 disparity map
- `sequence_id`, `image_id`, `fx`, `baseline`: metadata carried into manifests and evaluation records

The loader converts the stereo images to grayscale and uses the disparity array directly as float32 ground truth. Pixels with negative disparity are treated as invalid.

Why `mini_kitti` is used:

- it is the real-domain dataset in the current pipeline
- it is small enough to iterate quickly during fine-tuning and evaluation
- it provides clean stereo/disparity supervision in a Hugging Face-friendly format

In the teacher-student run, `mini_kitti` is used for:

- teacher fine-tuning on real data
- student fine-tuning and distillation on real data
- final evaluation and visualization

### Scene Flow pretraining

The pretraining source is `olivermao/sceneflow` on Hugging Face.

The repository exposes two tar archives:

- `frames_clean.tar`
- `disparity.tar`

The loader indexes left/right PNG image pairs in the frame archive and matches them with left-view `.pfm` disparity files in the disparity archive. The implementation reconstructs Scene Flow examples from the archive paths and decodes the disparity maps with a PFM reader.

Why Scene Flow is used:

- it is the large synthetic source with dense disparity labels
- it provides much more diversity than `mini_kitti`
- it is used to give both teacher and student broad patch-matching supervision before real-domain adaptation

In the teacher-student run, Scene Flow is used for:

- large-scale teacher pretraining
- large-scale student distillation pretraining

## Training flow

The end-to-end training entry point is `v2/training/train_real_stereo.py`.

### Stage 1: Scene Flow teacher pretraining

1. download or reuse the Hugging Face Scene Flow archive files
2. build a manifest over all available stereo pairs
3. iterate through the manifest in chunks
4. sample patch pairs from each chunk
5. train the larger teacher on the full synthetic corpus

### Stage 2: KITTI teacher fine-tuning

1. stream all available `mini_kitti` train examples
2. decode stereo images and zipped disparity arrays
3. sample patch pairs from valid disparity locations
4. fine-tune the teacher on the real-domain dataset

### Stage 3: Scene Flow student distillation

1. iterate through the full Scene Flow manifest again in chunks
2. sample patch pairs from each chunk
3. score each patch pair with the trained teacher
4. mix hard labels and teacher soft targets
5. train the smaller student on the distilled targets

### Stage 4: KITTI student fine-tuning

1. build all `mini_kitti` train patches
2. compute teacher scores for those patches
3. mix hard labels and teacher soft targets
4. fine-tune the student on the real-domain data

### Stage 5: validation and full disparity reconstruction

For each validation image:

1. convert left and right images to grayscale
2. build 5x5 windows over the full image
3. for each disparity `d` in `0..15`, pair the left patch with the right patch shifted by `d`
4. run the student model on those candidate pairs
5. stack the candidate scores into a score volume
6. choose the disparity with the highest score at each valid location

This produces a real-image disparity map output rather than a `.hex` stream.

### Stage 6: export artifacts

Each teacher-student run writes:

- timestamped artifacts under `logs/<timestamp>_<run_name>/`
- a saved teacher model (`teacher_patch_model.keras`)
- a saved Keras model (`student_patch_model.keras`)
- a parity batch for hls4ml checks (`parity_batch.npz`)
- sampled manifests for the run
- training history
- teacher and student metrics
- a training input grid (`training_input_grid.png`)
- a teacher/student result grid (`student_teacher_results.png`)

## Patch labeling strategy

The student is trained as a binary candidate matcher.

For each sampled valid pixel:

- positive sample: left patch with the right patch at the true disparity
- negative samples: the same left patch paired with one or more wrong disparities

During distillation, the hard labels are mixed with teacher probabilities. This keeps the learning problem aligned with the deployment boundary while still letting the student absorb the teacher's softer ranking signal.

## hls4ml path

The hls4ml entry point is `v2/hls4ml/convert_student.py`.

Flow:

1. load the trained Keras model
2. create an hls4ml config with `config_from_keras_model`
3. convert with the `Vitis` backend and `io_stream`
4. compile the generated hls4ml model in software
5. run parity checks against a saved batch of patch inputs

The current validated parity result from the smoke run was very close:

- mean absolute error: about `0.0032`
- max absolute error: about `0.0069`

That is the important first milestone before any FPGA synthesis work.

## Why native Keras instead of ONNX first

The student is authored directly in Keras, so native Keras is the most direct path into hls4ml.

Reasons:

- fewer moving parts than exporting to ONNX/QONNX first
- no extra graph-cleaning or channels-last conversion step
- easier debugging when training and conversion operate on the same model definition
- the architecture already fits inside hls4ml-supported Keras layers

ONNX or QONNX still makes sense later if the training stack moves to PyTorch or a quantized FINN-style flow, but it is not the shortest path for this student model.

## Current limitations

- the current teacher is still patch-based rather than a large whole-image stereo model
- full Scene Flow runs are heavier because the Hugging Face source is packaged as large tar archives
- the student currently uses standard floating-point Keras layers; quantized variants can follow once baseline accuracy and conversion stability are in place
- the final student is still limited by the fixed `0..15` disparity search assumption from the hardware-facing interface

## Next model-level steps

1. increase Scene Flow pretraining coverage with cached manifests and larger subsets
2. fine-tune on a larger KITTI split rather than the minimal smoke subset
3. add teacher pseudo-labeling or distillation
4. explore quantized Keras variants once the accuracy baseline is stable
5. map the exported student interface directly onto the future HDL replacement boundary