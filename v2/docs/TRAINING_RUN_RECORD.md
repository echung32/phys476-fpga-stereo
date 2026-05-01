# Training Run Record

This file records the main stereo training runs kept under `logs/` during the
final 16:9 tuning cycle, along with the HLS export decision.

## Current Best

- Best training run: `monkaa_crop16x9_sf25_fc32_h160_w288_bs128`
- Best HLS export candidate: `fixed<16,6>` feature extractor export from that run
- Reason: `fc32` improved all held-out metrics over `fc24`, while `fc40` failed
  with CUDA OOM at batch size `128`

## Comparison Table

| Run | Status | Shape | Fit | FC | SF frac | Best val MAE | Train driving holdout MAE | Train Scene Flow holdout MAE | Eval driving MAE | Eval mini-KITTI MAE | Eval Scene Flow MAE | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `monkaa_resize_h128_w384_d64_fc24_bs128` | complete | `128x384` | pad-like resize path from earlier run | 24 | 0.15 | 5.357 | 5.090 | 12.824 | not saved locally | not saved locally | not saved locally | Older wide baseline before 16:9 crop switch |
| `monkaa_crop16x9_sf15_h160_w288_bs128` | complete | `160x288` | crop | 24 | 0.15 | 5.246 | 4.308 | 7.633 | 3.686 | 4.333 | 7.606 | 16:9 crop improved geometry, but synthetic mix still too light |
| `monkaa_crop16x9_sf25_h160_w288_bs128` | complete | `160x288` | crop | 24 | 0.25 | 4.412 | 3.676 | 6.412 | 3.343 | 4.000 | 6.374 | First strong 16:9 winner; promoted before capacity sweep |
| `monkaa_crop16x9_sf25_fc32_h160_w288_bs128` | complete | `160x288` | crop | 32 | 0.25 | 3.677 | 3.443 | 5.227 | 3.103 | 3.392 | 5.201 | Current best overall |
| `monkaa_crop16x9_sf25_fc40_h160_w288_bs128` | failed | `160x288` | crop | 40 | 0.25 | 6.309* | 5.864* | 8.883* | n/a | n/a | n/a | OOM during epoch 4 at batch size `128`; values marked `*` are interim bests only |

## Notes

- The two intermediate `monkaa_ar_resample_*_h128_w384_*_v2` relaunches were
  intentionally stopped early when the pipeline moved from padded fitting to
  native 16:9 crop fitting. They are not ranked here because they were not
  final runs.
- The `fc32` run is the best checkpoint to carry forward into deployment work.
- The `fc40` run showed that more capacity is not free at the current batch
  size and memory budget.

## HLS Export Decision

Exports from the winning `fc32` checkpoint:

| Export | Precision | Mean abs parity error | Max abs parity error | Recommendation |
| --- | --- | ---: | ---: | --- |
| `fixed16_6` | `fixed<16,6>` | 0.0214 | 0.4352 | Use this for synthesis work |
| `fixed12_4` | `fixed<12,4>` | 0.0873 | 1.3072 | Too much extra drift relative to `fixed16_6` |

The `fixed<16,6>` export is the safer quantized choice to carry into HLS and
further hardware evaluation.
