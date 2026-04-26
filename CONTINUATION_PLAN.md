# Continuation Plan (Primary Path, Teacher Removed)

## Objective

Deliver an efficient, deployable stereo model quickly on the Artix-class target by removing teacher scope and moving to a low-resolution full-frame student architecture.

Chosen direction:

- remove teacher and distillation scope completely
- remove dead teacher-related code and documentation
- train one student model directly from ground-truth disparity
- use larger, mixed datasets and stereo-safe augmentations
- replace patch-MLP-first strategy with low-resolution shared-feature correlation

## Non-Negotiable Decisions

1. Teacher scope is fully removed from roadmap and codebase.
2. Student-only supervision with GT disparity is the only training path.
3. Primary model architecture is low-resolution full-frame correlation, not patch-only MLP.
4. One mixed-data training pass is the default training workflow.

## Why the Architecture Pivot Is Necessary

The prior patch-only plan (including 11x11 patch increase) was a minimal-risk incremental step, but it is not the best efficiency/quality trade for the board and timeline.

The primary architecture is now:

1. Downscale stereo input to 320x96 or 256x128.
2. Shared tiny feature extractor on left and right views (3-4 Conv2D layers, 8-16 channels).
3. Correlation/cost computation across a limited disparity range at reduced scale (24-48).
4. Winner-take-all or tiny refinement head.
5. Optional upsample/refine to output disparity resolution.

This better matches efficient downstream hardware than repeated per-pixel patch scoring while providing much more scene context.

## Final Model Strategy

## Primary Model (Implement)

- low-resolution full-frame student with shared left/right feature extractor
- disparity from correlation-based matching
- GT-only training

## Patch Pipeline Status

- patch pipeline is no longer the primary target
- keep only temporarily as baseline during migration
- remove once primary model reaches functional parity on validation/evaluation scripts

## Dataset Plan (Selected, Not Optional)

## Must-Use for Training

1. Scene Flow full set (FlyingThings3D, Driving, Monkaa)
   - dense synthetic GT and high geometric diversity

2. DrivingStereo
   - very large real-world driving corpus (about 174k training frames)

3. KITTI Stereo 2012 + KITTI Stereo 2015 training sets
   - real-domain benchmark supervision

## Evaluation-Only Initially

4. Middlebury
5. ETH3D two-view

These are reserved for held-out generalization checks, not immediate training integration.

## One-Pass Mixed Training

Recommended effective sampling ratio in training:

- 50% Scene Flow
- 35% DrivingStereo
- 15% KITTI 2012/2015 combined

This keeps synthetic diversity while continuously anchoring optimization on real data.

## Stereo-Safe Augmentation Policy

All augmentations must preserve stereo geometry and disparity consistency.

## Allowed

1. Shared crop (same crop for left, right, disparity, valid mask)
2. Shared resize with disparity scaling by horizontal scale factor
3. Shared pad/center-crop for fixed tensor sizes
4. Shared photometric jitter (brightness/contrast/gamma)
5. Shared mild noise/blur/compression simulation

## Forbidden

1. Horizontal flips
2. Independent left/right geometric transforms
3. Rotations that violate rectified epipolar geometry
4. Vertical stereo offsets
5. Any transform that changes pixel geometry without disparity-map remap

## Code Removal Plan (Dead Code Elimination)

Teacher scope is removed, so the following files must be deleted from active codebase:

1. v2/training/train_teacher_student.py
2. v2/models/keras_teacher.py

Likely removable after import check:

3. v2/training/visualization.py
   - remove if no longer imported by active student/evaluation paths

Documentation cleanup required:

4. v2/MODEL.md
   - remove teacher/distillation/FoundationStereo sections and references
5. README.md and any v2 docs
   - remove teacher-stage commands and references

Log and artifact naming cleanup:

6. remove expectations of teacher artifacts such as teacher_patch_model.keras from docs and run checklists

## Required Implementation Work (Primary Path)

1. Add/finish dataset adapters for DrivingStereo and KITTI 2012/2015.
2. Add new student model entrypoint for low-resolution shared-feature correlation.
3. Add mixed-dataset sampler for one-pass training.
4. Add stereo-safe augmentation module with strict policy above.
5. Keep hls4ml export path focused on the final student model only.

## Training Run Spec (Primary)

- model: low-resolution shared-feature correlation student
- supervision: GT only
- datasets: Scene Flow + DrivingStereo + KITTI 2012/2015
- no teacher checkpoints
- no pseudo-label generation
- no distillation targets

## Success Criteria

1. Training uses large mixed manifests (not mini-kitti-only).
2. End-to-end run is stable with no invalid-sample crashes.
3. Validation metrics exceed current mini_kitti-only baseline.
4. hls4ml conversion/parity remains functional for student export.

## Risks and Mitigations

1. Dataset heterogeneity
   - mitigation: isolate dataset parsers and normalize into one internal stereo example format.

2. Integration complexity of new architecture
   - mitigation: fix input size early (320x96 or 256x128) and keep model channels low (8-16).

3. Augmentation geometry mistakes
   - mitigation: centralize transforms and enforce forbidden-list checks.

4. Temporary duplication with legacy patch path
   - mitigation: time-box legacy baseline support and delete patch-primary training path once parity milestone is reached.

## Scope

In scope:

- student-only GT training
- low-resolution correlation architecture
- dataset expansion and stereo-safe augmentation
- hls4ml student export
- removal of teacher code and references

Out of scope:

- teacher models, distillation, pseudo-labeling
- FoundationStereo integration

## Immediate Deliverables

1. Dead-code removal PR scope list finalized and executed for teacher files.
2. New student architecture implementation for low-resolution correlation path.
3. Mixed-data training config and manifests for Scene Flow + DrivingStereo + KITTI.
4. Updated docs reflecting teacher-free architecture.
5. New student checkpoint + metrics + hls4ml parity artifacts.

## Bottom Line

The project now commits to a single efficient path:

- no teacher
- no dead teacher code
- low-resolution full-frame correlation student
- large GT data + stereo-safe augmentation

This is the fastest path to a strong deployable result.
