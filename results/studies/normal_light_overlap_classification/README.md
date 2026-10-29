# Normal-light source classification

All rated scenes use the same normal-light thumbnail test and fixed threshold. Classification was committed before model subgroup metrics were recomputed. The earlier check is preserved as history in the sibling normal_light_overlap_validation directory.

Run from the repository root:

```text
python -m analysis.studies.normal_overlap_classify --square-normal-dir <SQUARE-LOL>/all_normal --lol-normal-dir <LOL>/our485/high --lowlight-labels results/studies/normal_light_overlap_validation/lowlight_labels.csv --output <new-audit-directory> --classification-record <new-classification.csv>
```

PLAN.json and its SHA-256 precede the image calculation. results.json contains the classification decision, image hashes and correlation separation; classification.csv contains all scene matches. No images or model outputs are included in this decision record.

## Source-overlap record correction

The matching population is the 290 rated SQUARE-LOL scenes. The original lowlight archive includes unrated low00416 but omits low00710, which has ten rated outputs. All 289 available analysis-source thumbnail matches were reproduced with the original equalized 48 by 32 grayscale correlation and threshold 0.9. At that stage, the record contained 229 shared scenes, 60 verified non-overlapping scenes, and one scene awaiting image evidence. Missing input is not evidence of non-overlap. The previous 61-scene complement combined verified non-overlap with low00710; its figures remain in the historical JSON, marked as superseded. The source archive and MOS list provide no per-scene collection identifiers. `results/studies/overlap_transfer_verified.json` and `records/squarelol_lol_scene_matches.csv` record the correction; existing predictions were regrouped without fitting or inference.




### Normal-light fallback validation

Before classifying the missing low-light source, the same equalized 48 by 32 grayscale-thumbnail correlation and fixed 0.9 threshold were applied to normal-light references for all 289 scenes with known low-light classifications, against the LOL our485 normal-light images. The shared/non-overlap decisions agreed for 288/289 scenes (99.653979%). The sole disagreement was low00182: normal-light correlation 0.999677739 with 186.png, versus recorded low-light correlation 0.8729 with 185.png.

The prespecified criterion required complete class agreement. It was not met, so low00710 was left unresolved and the 229 shared / 60 verified non-overlapping / one unresolved split was retained at that stage. This decision is superseded by the complete normal-light classification below. No threshold was adjusted and no image model was rerun. The plan, per-scene validation, input-label snapshot and image hashes are in `results/studies/normal_light_overlap_validation/`; the reproduction script is `analysis/studies/normal_overlap_check.py`.

### Complete normal-light classification

The complete-agreement check treated the low-light labels as ground truth, although dark images can give noisy thumbnail matches. Image evidence identifies its sole discrepancy, low00182, as a shared scene: its normal-light reference matches LOL 186.png at 0.999677739, while its low-light match selected adjacent LOL frame 185.png at 0.8729. The decision to use normal-light references for every scene was made before examining or recomputing any model subgroup metric. No correlation threshold or image transformation was tuned.

All 290 rated scenes were rechecked using the original equalized 48 by 32 grayscale-thumbnail correlation and fixed threshold 0.9. The lowest shared-scene correlation is 0.989475467; the highest non-overlapping correlation is 0.731615159. low00710 matches 710.png at 1.000000000 and is shared. The final partition is 231 shared and 59 non-overlapping scenes, with no scene left unresolved. Low-light matching agreed on 288 of its 289 available scenes and missed one shared scene.

The plan, decision, every match and image hashes are preserved in `results/studies/normal_light_overlap_classification/`. `analysis/studies/overlap_regroup.py` regrouped the same fixed per-image predictions without training or model inference. `results/studies/overlap_transfer_verified.json` and `results/lmm_extra/overlap_verified.json` are the current summaries; the prior validation plan and failed check remain in `normal_light_overlap_validation/`.
