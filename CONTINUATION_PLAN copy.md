# Continuation Plan

## Objective

Resume the `v2` stereo pipeline with the correct teacher design:

- keep the deployable model as the compact Keras student used for hls4ml
- stop treating the scratch patch teacher as the primary path
- integrate a pretrained whole-image teacher, with FoundationStereo as the main target
- use the teacher offline to generate cached pseudo-labels or distilled supervision for the student

## Current Repository State

### Completed

- `uv` environment is in place and the repo is using `uv run ...`
- the legacy baseline has been moved under `v1/`
- the `v1` `.hex` visualizer and baseline file layout were repaired
- `v2` has a working real-data Keras student pipeline using `KERAS_BACKEND=torch`
- `mini_kitti` loading works
- `olivermao/sceneflow` tar-archive loading works
- Keras student to hls4ml conversion works in software
- smoke runs succeeded for the current student flow and for the temporary teacher-student flow
- timestamped run directories under `logs/` work
- training input and result-grid visualizations are being written into run directories

### Important Current Files

- `v2/training/train_real_stereo.py`: validated student-only training path
- `v2/training/train_teacher_student.py`: current temporary teacher-student orchestration
- `v2/training/stereo_data.py`: Hugging Face and tar-archive dataset loading
- `v2/training/patch_dataset.py`: patch extraction, candidate generation, reconstruction, metrics
- `v2/models/keras_student.py`: deployable student
- `v2/models/keras_teacher.py`: temporary scratch teacher; should not remain the main teacher path
- `v2/hls4ml/convert_student.py`: validated hls4ml export/parity path
- `v2/MODEL.md`: documentation, but it should be revised again once FoundationStereo replaces the scratch teacher in the main design

## Latest Findings

### GPU Behavior

The current long run was not failing because CUDA was unavailable.

Evidence:

- the live log reached `teacher_sceneflow chunk 1/7`, so model training did start
- GPU memory moved upward during launch
- the major bottleneck before and between chunks is CPU-side work

The current script is CPU-heavy because it does all of the following synchronously:

- tar member access from `frames_clean.tar` and `disparity.tar`
- PNG decode through PIL
- PFM decode on CPU
- grayscale conversion on CPU
- patch extraction and negative sampling with NumPy and Python loops
- chunk construction before each `fit()` call

This means the GPU can sit underutilized while the next Scene Flow chunk is being prepared.

### Current Failure in the Long Run

The full run under `logs/20260421_160504_full_teacher_student_gpu1/` failed after the first Scene Flow chunk with:

`ValueError: No valid training positions were found for the selected disparity range`

Root cause:

- some Scene Flow samples contain no valid pixels after applying the current constraints in `sample_training_examples()`
- the current training loop assumes every loaded Scene Flow sample yields valid patch positions

This is a local bug in the current temporary teacher-student path. It is fixable, but it should not be the primary direction if the teacher is being replaced by FoundationStereo.

### FoundationStereo Clarification

FoundationStereo is not a self-contained weight blob that can be used without model code.

The released checkpoint provides learned parameters, but inference still requires:

- the FoundationStereo model definition
- the matching config file such as `cfg.yaml`
- the same preprocessing and inference conventions expected by that model

So the correct interpretation is:

- the checkpoint means we do not need to train the teacher from scratch
- but we still need the upstream FoundationStereo inference code or a minimal local wrapper around the same architecture

## Strategic Decision

The main teacher path should pivot to FoundationStereo.

### Primary Design Going Forward

1. Use pretrained FoundationStereo for offline dense disparity inference.
2. Cache teacher outputs on the stereo datasets used by this repo.
3. Train the current compact Keras student from those teacher outputs plus available ground truth.
4. Keep hls4ml focused on the student only.

### Secondary / Fallback Path

The scratch patch teacher may remain only for:

- smoke testing
- ablation experiments
- fallback if FoundationStereo integration becomes blocked

It should not remain the main narrative or default training flow.

## Recommended Environment Split

Do not force FoundationStereo into the exact same environment unless it installs cleanly with low friction.

Recommended approach:

### Existing `uv` environment

Use this for:

- dataset loading
- patch extraction
- Keras student training
- hls4ml conversion
- plotting and evaluation

### Separate FoundationStereo inference environment

Use this for:

- checkpoint download
- FoundationStereo inference
- cached teacher disparity generation

Rationale:

- FoundationStereo has its own dependency surface and CUDA expectations
- isolating it reduces the risk of destabilizing the Keras+hls4ml stack
- the clean interface between the two systems is cached disparity output, not shared training code

## FoundationStereo Integration Plan

### Phase 1: Bring in inference-only teacher support

Goal:

- run pretrained FoundationStereo on stereo pairs from this repo

Tasks:

1. Add a dedicated area under `v2/teachers/foundationstereo/` or a separate external checkout path.
2. Download the released checkpoint folder using `gdown` if needed.
3. Keep the entire checkpoint folder structure, not only the `.pth`, because the config is also needed.
4. Validate single-pair inference on one sample from `mini_kitti`.
5. Save the teacher disparity map and a visualization for inspection.

Expected outcome:

- one validated offline teacher inference path

### Phase 2: Cache teacher outputs

Goal:

- make teacher inference a preprocessing step instead of an online dependency during student training

Tasks:

1. Add a script that iterates over selected dataset examples.
2. For each example, run FoundationStereo once.
3. Save outputs as local artifacts, for example `.npz` files containing:
   - teacher disparity
   - image id / sequence id metadata
   - optional confidence or mask if a usable confidence proxy is available
4. Build a manifest mapping raw stereo pairs to cached teacher outputs.

Expected outcome:

- repeated student runs do not need to invoke FoundationStereo live

### Phase 3: Adapt student supervision to teacher outputs

Goal:

- teach the patch student from a whole-image disparity teacher

Tasks:

1. Extend patch extraction so it can sample labels from cached teacher disparity maps.
2. Define one supervision mode at a time:
   - first: teacher disparity as pseudo-ground-truth for candidate labels
   - later: confidence-weighted distillation if useful
3. Keep the student input and output unchanged:
   - input: 5x5 left/right patch pair
   - output: scalar match score for one disparity candidate
4. Reconstruct disparity maps exactly as before with the external `0..15` search loop.

Expected outcome:

- student remains hls4ml-friendly while teacher provides stronger supervision

### Phase 4: Re-run documentation and evaluation

Goal:

- align docs and metrics with the real teacher design

Tasks:

1. Update `v2/MODEL.md` to describe FoundationStereo as the primary teacher.
2. Remove wording that suggests the scratch patch teacher is the main intended teacher.
3. Re-run evaluation artifacts and save teacher-vs-student result figures.

## Concrete Next Session Steps

When resuming, do the following in order.

### Step 1: Stop investing in the scratch teacher as the main path

- do not launch more long runs with the current temporary teacher-student script unless needed for debugging a fallback path
- treat the existing script as provisional

### Step 2: Add checkpoint download support

If `gdown` is not already present, add it with `uv add gdown` only if the download helper is intended to live in the repo environment.

Recommended checkpoint workflow:

1. create a target directory such as `v2/teachers/foundationstereo/pretrained_models/`
2. download the checkpoint folder contents from the provided Google Drive source
3. verify that both model weights and config files are present

### Step 3: Validate FoundationStereo on one stereo pair

Success criteria:

- one `mini_kitti` stereo pair runs through the pretrained teacher
- a dense disparity map is produced
- output is written to a local artifact directory

### Step 4: Add cached teacher-output generation

Start with a very small subset:

- 4 to 8 `mini_kitti` train examples
- 4 validation examples

Generate and save:

- teacher disparity `.npz`
- visualization PNGs
- manifest JSON linking dataset metadata to teacher artifacts

### Step 5: Refactor student training to consume cached teacher outputs

This is the real replacement for the current scratch-teacher loop.

Target behavior:

- no online teacher training stage
- no online teacher prediction stage during each student batch
- training reads cached teacher maps and samples patch labels from them

### Step 6: Keep the student export path unchanged

The hls4ml path should stay the same:

- train student
- save Keras model
- run `v2/hls4ml/convert_student.py`
- compare parity metrics

## If the Scratch Teacher Path Must Be Reused Briefly

If there is a need to keep the current temporary path running for comparison, fix these local issues first:

1. In `build_training_examples()` or the caller, skip Scene Flow samples that yield no valid positions instead of failing the whole chunk.
2. Add timing logs around:
   - Scene Flow load time
   - patch-build time
   - `fit()` time
3. Consider prefetching the next chunk while the current chunk trains.
4. Consider vectorizing parts of patch sampling later if throughput becomes important.

This is a fallback track only.

## Recommended Deliverables for the Next Work Session

The next session should aim to finish these concrete outputs:

1. a FoundationStereo inference path that runs on one example
2. a local cache format for teacher disparity outputs
3. a small cached-teacher manifest for `mini_kitti`
4. a revised student-training path that can consume cached teacher outputs
5. updated `v2/MODEL.md` reflecting the new primary teacher design

## Commands and Workflow Notes

- Use `uv run python ...` for repo-managed Python entry points.
- Keep `KERAS_BACKEND=torch` for the Keras student and hls4ml flow.
- The validated student stack is separate from the likely FoundationStereo dependency stack.
- The current long run artifacts live under `logs/` and can be used as reference outputs, but they should not define the final architecture direction.

## Summary

The repo is already far enough along that the student, dataset loaders, logging, visualization, and hls4ml software path are working.

The main unfinished architectural correction is the teacher path:

- the project should use a pretrained whole-image teacher such as FoundationStereo
- teacher inference should be offline and cached
- the compact student should remain the only deployment target

That is the correct continuation path for the next session.