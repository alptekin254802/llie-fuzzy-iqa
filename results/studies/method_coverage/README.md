# Human calibration with broader method coverage

This exploratory control keeps four test methods fixed within each repeat and adds training methods and their images in nested sets of two through six. It extends Supplement S14; the distributed LOL model is unchanged. Numeric summaries are in [REPORT.md](REPORT.md), with the complete split design in config.json and method_splits.csv.

Verify the shipped numeric summaries without datasets:

```bash
python -B -m analysis.method_coverage summary
```

For full reproduction, obtain SQUARE-LOL, LIEQ and RLIE images and scores from their providers, configure study_config.local.json, and prepare the area256 feature and BRISQUE caches as described in [STUDIES.md](../../../docs/STUDIES.md#data-layout-and-preparation). MOS values are read from these locally prepared caches; no LIEQ or RLIE image-level scores are distributed here. Use a fresh output directory:

```bash
python -B -m analysis.method_coverage prepare --config study_config.local.json --out study_work/method_coverage_reproduction
python -B -m analysis.method_coverage run --config study_config.local.json --out study_work/method_coverage_reproduction
python -B -m analysis.method_coverage analyze --config study_config.local.json --out study_work/method_coverage_reproduction
python -B -m analysis.method_coverage verify --config study_config.local.json --out study_work/method_coverage_reproduction
```

Preparation checks the original S14 five-versus-five results before fixing the new splits. Verification independently checks normal equations, predictions, ranks, scene/method separation and bootstrap draws. Local predictions, fitted parameters, bootstrap arrays and plots are generated during reproduction and are not distributed. Intervals resample scenes with the 20 selected method splits and fitted models held fixed, and have no multiplicity adjustment.
