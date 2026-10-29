# Normal-light fallback validation

Class agreement: 288/289 (99.653979%). The preregistered complete-agreement criterion was not met; low00710 was left unresolved at that stage. This historical check is superseded by [complete normal-light classification](../normal_light_overlap_classification/README.md). The discrepancy is low00182 (normal-light 0.999677739; low-light 0.8729; threshold 0.9).

Run from the repository root, using the original dataset folders and a new output directory:

```text
python -m analysis.studies.normal_overlap_check --square-normal-dir <SQUARE-LOL>/all_normal --lol-normal-dir <LOL>/our485/high --matches results/studies/normal_light_overlap_validation/lowlight_labels.csv --output <new-audit-directory>
```

`PLAN.json` was saved before the correlations were computed. `known_scene_validation.csv` contains every known-scene result. `results.json` contains the decision and source-image hashes; no images are distributed.
