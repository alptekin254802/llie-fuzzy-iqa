# Reproducing the validation studies

Copy `study_config.example.json` to a local configuration file. All paths are
relative to that file unless absolute. `work_dir` must be separate from the shipped
`results/` and `models/` directories. Use a new work directory when changing data
roots or numerical protocols. The runner records a configuration fingerprint.

Every command starts with:

```bash
python -B -m analysis.studies --config study_config.local.json
```

Append the study name and arguments below. Computationally expensive commands are
explicit: none are run by the test suite. Image features are cached in the work
directory, with resolution in the filename. Feature and BRISQUE extraction accept
a wall-time budget in seconds; repeat the same command until all rows are present.
Use `36000` to allow a complete extraction. Do not fit a study on a partial cache.

## Data layout and preparation

Obtain the original images and labels from their providers. They are not included
in this code repository. Configure:

| Configuration key | Expected contents |
|---|---|
| `lol_root` | LOL `our485/low` and `our485/high`, matching PNG names |
| `preprocessed` | Bilinear 256-square LOL `low` and `high`, created by `preprocessing.preprocess` |
| `squarelol` | `name_mos.txt`, `enhancement/`, `lowlight/`, `all_normal/` |
| `lieq` | `MOS.mat`, `imgNames.mat`, `dataset/Enhancement/`, `dataset/Reference/` |
| `rlie` | `normalized_scores.csv`, `enhancement/`; no reference images required |

Regenerate classical metric inputs with `python -m evaluation.classical_metrics --preprocessed /path/to/preprocessed --output new_metrics.csv`.
`python -m analysis.validate_handtuned` verifies the fixed reference-based baseline against all stored scores.

Run `prepare` once to assemble keyed LOL validation inputs from the included
derived CSVs. Run `features squarelol area256 36000`, and repeat for `lieq` and
`rlie`. The human-study images use OpenCV INTER_AREA. For the native study repeat
with `native` instead of `area256`. BRISQUE studies additionally require
`brisque-features DATASET 36000 area256` for each human set, and
`lol-brisque 36000` for enhanced LOL images.

## Paper-to-command map

| Paper destination | Study / entry point | Principal JSON output |
|---|---|---|
| Methods; main model and Table 1 | `canonical-human`; `analysis.validate_monotone` | `canonical_human.json`; `results/monotone_validation.json` |
| Section S3, partition control | `python -m analysis.monotone_partition_control` | `results/monotone_partition_control.json` |
| Main Table 3 and Section S5, scene ranking and method-mean comparison | `method-mean` | `method_mean.json` |
| Table S8, within-method across-scene correlation | `python -m analysis.make_figures` | `results/figures_numbers.csv` |
| Table S10, capacity ablation | `capacity-ablation` | `capacity_ablation.json` |
| Section S13, human transfer and individual features | `human-transfer DATASET area256`; `canonical-human` | `human_transfer_DATASET_area256.json`; `canonical_human.json` |
| Table S14, 27 DE settings | `de-sensitivity` | `de_sensitivity.json` |
| Section S10, SSR diagnosis | `retinex` after `regional` feature extraction | `retinex.json` |
| Table S15, regional features | `regional` | `regional.json` |
| Table S15a, exploratory unseen enhancement methods | `python -B -m analysis.unseen_operators verify` | [`results.csv`](../results/studies/unseen_operators/results.csv); [protocol and reproduction](../results/studies/unseen_operators/README.md) |
| Table S16, complete native pipeline | `native` | `native_clean.json` |
| Table S18, live candidate search | `python -B -m realtime.summarize` | `results/live_timing/mjpg_30fps/summary.json` |
| Table S17, synthetic exposure selection | `closed-loop` | `closed_loop.json` |
| Section S13, pretrained metrics | `human-metrics`; then `human-metric-summary` | `human_metrics_analysis.json` |
| Table S11, BRISQUE-feature proxy regressor | `proxy-brisque` | `proxy_brisque.json` |
| Human transfer uncertainty | `transfer-uncertainty` | `transfer_uncertainty.json` |
| SQUARE-LOL overlap sensitivity | `normal_overlap_classify`; commit the image-based classification; then `overlap_regroup --classification-commit <commit>` (both via `python -m analysis.studies.<module>`) | [`overlap_transfer_verified.json`](../results/studies/overlap_transfer_verified.json) |
| Table S22, human recalibration | `human-recalibration DATASET` | `human_recalibration_DATASET.json` |
| Table S23, unseen methods | `human-method-disjoint DATASET` | `human_method_disjoint_DATASET.json` |
| Supplementary monotonicity note | `monotonicity` | `monotonicity.json` |
| Unsupported-rule discussion | `rule-support` | `rule_support.json` |

For the current overlap protocol, see the [normal-light classification instructions](../results/studies/normal_light_overlap_classification/README.md). The earlier `overlap_transfer.json` is a historical low-light partition and is not the current sensitivity result.

`DATASET` means each of `squarelol`, `lieq`, and `rlie`. The six-feature RLIE
calibration uses `python -m analysis.rlie_enriched --rlie_root /path/to/RLIE`.
The seven-method calibration is reproduced by `python -m analysis.expanded_validation`.
Pretrained-metric paired intervals and proxy method preferences use
`human-metric-uncertainty` (2,000 draws, seed 7).
The original fusion, disjoint-operator, unpaired-transfer and preprocessing
controls retain their entry points listed in the main README. Broader calibration
uses `python -m analysis.tier3_broaden` for the three reported operators.

The training-fold method-mean baseline is constant for a given method within
each test fold. Pooled within-method correlations can reflect differences between
fold means and do not demonstrate scene-specific ranking.

The frozen deployment models and main CV models are distinct fits. The native
and closed-loop controls refit all preprocessing components within their training
sets and must not substitute for the main-protocol CV result. NR-mono main CV is
canonical under one BLAS thread, the pinned numerical environment and exact
floating-point ranks. Full predictions and parameters are saved. Rounding a
near-tied prediction can change a rank; `near_tie_sensitivity` reports this separately.

## Additional quality metrics and score direction

Provide unmodified author releases externally. `bpri_source` and `bmpri_source`
point to folders containing `BPRI.m` and `BMPRI.m`, respectively, with their released
model/range files and LIBSVM helpers. `matlab` is the MATLAB executable. The adapter
copies these folders to its work directory because the author implementations
write temporary files. It does not retrain the models. `stairiqa_source` points to
the [StairIQA author repository](https://github.com/sunwei925/StairIQA), and
`stairiqa_weights` to the IMDT `ResNet_staircase_50-EXP1-Koniq10k.pkl` checkpoint.
It uses Resize(384), FiveCrop(320), ImageNet normalization and the KonIQ head
(index 3), averaged over crops.

Run `extra-metrics prepare`, `extra-metrics bpri`, `extra-metrics bmpri`,
`extra-metrics stairiqa`, then `extra-metrics analyze`. LOL comparisons use the
stored held-out NR-free predictions. MATLAB receives uint8 RGB, including an
explicit floor-quantization adapter for floating-point LOL enhancement outputs.
For isolated 50-image timing after preparation, run `extra-timing` on an idle machine.
Use the same power profile as the recorded campaign; setup does not change system power settings.

Directions are BPRI -1, BMPRI -1, StairIQA +1; they are not selected from test labels.

For the independent direction check use a **fresh configuration/work directory**:
`direction prepare`, `extra-metrics bpri`, `extra-metrics bmpri`, `direction analyze`.
Ten lexicographically first normal-exposure LOL references are resized with AREA;
each receives five noise, blur and JPEG levels plus its clean baseline. Noise uses
one BGR-order standard-normal field per scene (seed 20261018 + scene index), rounded
to nearest uint8; sigma is 5/10/20/30/50. Blur sigma is 0.5/1/2/3/5 with reflected
borders. JPEG quality is 90/70/50/30/10. Positive raw-score association with distortion
confirms lower-is-better. This is a direction check, not a validation of LLIE quality;
BPRI can respond oppositely at the mildest noise level.

RichIQA uses the released author weights on Hugging Face. See [RICHIQA.md](RICHIQA.md) for the unchanged official inference protocol,
pinned source and weight hashes, all four datasets, scene-bootstrap comparisons,
and separate CPU/GPU timings.

## Statistical and computational conventions

Scene bootstrap uses 2,000 draws, seed 20261018, fixed fitted predictions, percentile
intervals and finite-corrected null-centered p-values unless the specific transfer
uncertainty protocol states 5,000 draws and seed 11. Target-tied pairs are omitted;
prediction ties receive half credit; top-choice accuracy is the expectation under
uniform selection among maxima. Constant-scene SRCC is recorded as zero. The main
paired significance table uses Holm adjustment within each anchor. Ordinal-rank
optimization in the human within-scene DE objective is distinct from proxy fitting.

Regional variants retain the original decision rule: SSR gain at least 0.10 or
a strictly positive SQUARE-LOL paired interval, with LOL fold-mean loss below 0.02.
The additional paired uncertainty is descriptive follow-up, not independent
confirmation. SSR artifact associations are diagnostic, not causal.

The native comparison changes enhancement, features and LPIPS resolution together.
The closed-loop experiment uses synthetic exposures [1, .25, .5, .125], 100 held-out
scenes, gamma exponents .2 through 1.0, and default 2/3. Training-selected fixed
gamma uses training scenes only. Maxima within 1e-9 use continuity, then distance
to default, then the lower gamma; uniform-tie expectations are reported separately.
Its search time includes all nine candidates and is not a single-score latency.

## Archived results

`results/studies/` contains derived summaries of the reported campaigns; raw human
images and MOS are obtained separately. The main NR-mono record, frozen-human
evaluation, method-mean comparison, monotonicity check, controller aggregation and
direction aggregation, all three human-recalibration and method-disjoint studies,
and the seven-method calibration have complete current execution records. Other campaign
summaries preserve the recorded study values; their public implementations were
checked against the corresponding computational routines and representative inputs.
This release does not claim a new end-to-end GPU/MATLAB rerun of every campaign.

Classical external summaries with rounded display fields retain that display
precision. Full-precision predictions are generated in the work directory when
rerunning. Do not infer equivalence or select a model from last-digit rank changes.

## Saved derived records

`results/studies/records/` contains the closed-loop candidate and selector rows,
human-recalibration OOF predictions, scene-fold assignments, repeated method
splits, and the SQUARE-LOL/LOL best-scene matches with correlation scores.
These files contain no images. The two SQUARE-LOL frozen-prediction tables retain human scores under the MIT permission in [NOTICE](../NOTICE); LIEQ and RLIE per-image human scores are obtained separately. `manifest.json` records source
and output hashes and the exact split rules. The human scene-fold table applies
to both recalibration and scene-plus-method tests; combine it with each method
split, leaving the two unused cross-cells out of that test.

`closure_summary.json` reports tie-aware random Top-1 chance, the five recorded
LIEQ calibration seeds, and historical non-overlap scores computed from existing LPIPS-Alex
and Q-Align fp16 predictions. RF was not added to that subset: the stored RF
prediction files came from a different, human-trained experiment.

Use a fresh `work_dir` for new runs. The optional `archived_work_dir` field describes a local archive and does not affect computation.

To rebuild only regional decisions from a saved complete result and the nine
`study_regional_V{0,1,2}_de_{squarelol,lieq,rlie}_predictions.csv` files:

```bash
python -B -m analysis.studies.regional --summary-from regional.json --predictions saved_predictions --output regional_decisions.json
```

This command neither loads images nor fits models. A clean-directory check
reproduced all 15 displayed regional values and the recorded paired decisions;
see `results/studies/regional_summary_verification.json`.

## Stored-score anchor and demo checks

Run `python -B -m analysis.audit_derived` to recompute the seven added BRISQUE comparisons, the unpaired entropy control, the FR-calibrated/NR-free gap and the demo frame-change/score-change correlation. Inputs are existing numeric score records; outputs are `results/studies/audit_derived.json` and `audit_anchor_folds.csv`. No model fitting or images are required.

## Targeted calibration-source overlap control

[The targeted-removal control](../results/studies/targeted_overlap/README.md) refits NR-free without the 231 shared scenes and compares it with 20 equally sized random-removal fits. Run `python -B -m analysis.studies.targeted_overlap` to verify the stored summaries, or add `--out study_work/targeted_overlap_reproduction` to reproduce all fits. Source classification remains the verified normal-light classification; this study changes calibration membership only.

The [training-method coverage control](../results/studies/method_coverage/README.md) extends Supplement S14: `python -B -m analysis.method_coverage summary` verifies the shipped summaries, and the linked instructions reproduce fitting and scene-bootstrap contrasts.
