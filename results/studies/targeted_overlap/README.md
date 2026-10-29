# Calibration-source overlap control

This follow-up removes all 231 LOL scenes shared with SQUARE-LOL before fitting NR-free, leaving 254 scenes and 1,270 enhanced outputs. The positive control refits all 485 LOL scenes. Both frozen models are evaluated on the same 290 SQUARE-LOL scenes and the 59 scenes without LOL source overlap.

The random control removes 231 uniformly sampled scenes for each of seeds 1 through 20, independently of overlap labels. Calibration settings, DE seed 42, feature orientation, membership fitting and evaluation are identical. The ranges are minima and maxima across these 20 fits, not confidence intervals; no hypothesis test is performed. See [PROTOCOL.json](PROTOCOL.json) and [technical notes](../../../docs/TECHNICAL_NOTES.md#targeted-calibration-source-overlap-control).

Verify the stored summary without model fitting:

```sh
python -B -m analysis.studies.targeted_overlap
```

Reproduce the two calibration controls and all 20 random subsets in a new directory:

```sh
python -B -m analysis.studies.targeted_overlap --out study_work/targeted_overlap_reproduction
```

The script checks every reported correlation and subset hash against the archived CSVs, excluding runtime. The original fitting and scoring functions are imported unchanged. `squarelol_features.csv` contains the existing three features from 256 x 256 area-interpolated images and SQUARE-LOL scores; their MIT notice is in [NOTICE](../../../NOTICE). No images, weights or labels from LIEQ/RLIE are included. `sharpness` is the historical field name for normalised Laplacian energy.

`results.csv` records the positive and targeted controls; `random_subsets.csv` contains all repetitions; `random_subsets_summary.csv` records the mean, range and descriptive counts. The two saved models, scene split and per-scene correlations support the targeted control. `EXPORT.json` records distribution-file hashes.
